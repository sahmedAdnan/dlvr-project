import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import ResNet101_Weights
from PIL import Image, UnidentifiedImageError
import os
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

########################################
# Set Random Seeds for Reproducibility
########################################
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

########################################
# Device configuration
########################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################
# Initialize the BERT tokenizer
########################################
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Ensure pad_token is set properly
if tokenizer.pad_token is None:
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

########################################
# Dataset
########################################
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read the captions file
        self.captions = []
        self.imgs = []
        self.img_ids = []
        with open(captions_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split('\t')
                if len(tokens) == 2:
                    img_id, caption = tokens
                    img_id = img_id.split('#')[0]
                    self.imgs.append(img_id)
                    self.captions.append(caption)
                    self.img_ids.append(img_id)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.img_ids[index]
        img_path = os.path.join(self.root_dir, img_id)

        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, UnidentifiedImageError) as e:
            print(f"Cannot open image {img_path}: {e}")

            return None, None, None

        if self.transform is not None:
            image = self.transform(image)

        # Tokenize the caption using BERT tokenizer
        numericalized_caption = tokenizer.encode(
            caption,
            add_special_tokens=True,
            max_length=50,
            truncation=True
        )
        caption_tensor = torch.tensor(numericalized_caption)

        return image, caption_tensor, img_id

########################################
# Custom Collate Function
########################################
class MyCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch = [(img, cap, img_id) for img, cap, img_id in batch if img is not None and cap is not None]
        if len(batch) == 0:
            return None

        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)  # (B, 3, 224, 224)
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.pad_token_id)
        img_ids = [item[2] for item in batch]
        return imgs, captions, img_ids

########################################
# DataLoader Function
########################################
def get_loader(root_folder, annotation_file, transform, batch_size=64, shuffle=True, num_workers=4, pin_memory=True):
    dataset = Flickr8kDataset(root_folder, annotation_file, transform=transform)
    pad_token_id = tokenizer.pad_token_id

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_token_id=pad_token_id)
    )
    return loader, dataset

########################################
# Encoder with Spatial Feature Extraction
########################################
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Load ResNet-101 with pretrained weights
        resnet = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        # Remove the fully connected layers and the adaptive pool
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        # The shape of features: (batch, 2048, 7, 7) after ResNet-101
        self.conv2d = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.bn(self.conv2d(features))
        features = features.view(features.size(0), features.size(1), -1)
        # Transpose to (B, 49, embed_size)
        features = features.permute(0, 2, 1)
        return features

########################################
# Attention Module
########################################
class Attention(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_size + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, features, hidden):
        # Expand hidden to match features
        hidden = hidden.squeeze(0)
        hidden = hidden.unsqueeze(1).expand_as(features[:, :, :hidden.size(-1)])
        # Concatenate features and hidden
        concat = torch.cat([features, hidden], dim=2)
        energy = torch.tanh(self.attention(concat))
        attention = self.v(energy).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        # Weighted sum of features
        context = (features * alpha.unsqueeze(2)).sum(dim=1)  # (B, feature_size)
        return context, alpha

########################################
# Decoder with Attention
########################################
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers=num_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)
        self.attention = Attention(embed_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)

        h, c = self.init_hidden(features.size(0))
        outputs = []

        # One step at a time
        for t in range(embeddings.size(0)):
            # Apply attention
            context, alpha = self.attention(features, h)
            lstm_input = torch.cat([embeddings[t], context], dim=1).unsqueeze(0)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.linear(output.squeeze(0))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        return outputs

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size).to(device),
                torch.zeros(1, batch_size, self.hidden_size).to(device))

    def sample(self, features, max_len=20):

        sampled_ids = []
        h, c = self.init_hidden(features.size(0))
        # Start with the [CLS] token as the initial input
        inputs = self.embed(torch.tensor([tokenizer.cls_token_id]*features.size(0)).to(device))  # (B, embed_size)
        for _ in range(max_len):
            context, alpha = self.attention(features, h)  # (B, embed_size)
            lstm_input = torch.cat([inputs, context], dim=1).unsqueeze(0)  # (1, B, 2*embed_size)
            output, (h, c) = self.lstm(lstm_input, (h, c))  # output: (1, B, hidden_size)
            output = self.linear(output.squeeze(0))  # (B, vocab_size)
            _, predicted = output.max(1)  # (B)
            sampled_ids.append(predicted.cpu().numpy())
            inputs = self.embed(predicted)  # (B, embed_size)
            # If all batches have generated [SEP], stop
            if (predicted == tokenizer.sep_token_id).all():
                break
        # Flatten the list
        sampled_ids = [item for sublist in sampled_ids for item in sublist]
        return sampled_ids

    def beam_search(self, features, beam_width=3, max_len=20):

        # For simplicity, assume batch_size=1
        assert features.size(0) == 1, "Beam search is implemented for batch_size=1."

        features = features.squeeze(0)  # (num_pixels, embed_size)
        h, c = self.init_hidden(1)  # (1, 1, hidden_size)
        # Start with the [CLS] token
        start_token = torch.tensor([tokenizer.cls_token_id]).to(device)  # (1)
        inputs = self.embed(start_token)  # (1, embed_size)

        # Initialize the beam with the start token
        beams = [( [tokenizer.cls_token_id], 0.0, h, c, inputs )]  # (sequence, score, h, c, inputs)

        for _ in range(max_len):
            new_beams = []
            for seq, score, h, c, inputs in beams:
                # Apply attention
                context, alpha = self.attention(features.unsqueeze(0), h)  # (1, embed_size)
                # Concatenate context and input
                lstm_input = torch.cat([inputs, context], dim=1).unsqueeze(0)  # (1, 1, 2*embed_size)
                output, (h_new, c_new) = self.lstm(lstm_input, (h, c))  # output: (1, 1, hidden_size)
                output = self.linear(output.squeeze(0))  # (1, vocab_size)
                log_probs = torch.log_softmax(output, dim=1)  # (1, vocab_size)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=1)  # (1, beam_width)

                for i in range(beam_width):
                    token_id = top_indices[0, i].item()
                    token_log_prob = top_log_probs[0, i].item()
                    new_seq = seq + [token_id]
                    new_score = score + token_log_prob
                    new_inputs = self.embed(torch.tensor([token_id]).to(device))  # (1, embed_size)
                    new_beams.append( (new_seq, new_score, h_new, c_new, new_inputs) )

            # Select top beam_width beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Check if all beams have generated [SEP]
            end_count = sum([1 for beam in beams if beam[0][-1] == tokenizer.sep_token_id])
            if end_count == len(beams):
                break

        # Choose the beam with the highest score
        best_seq = beams[0][0]
        return best_seq

########################################
# Hyperparameters and Setup
########################################
embed_size = 512
hidden_size = 512
num_layers = 1
learning_rate = 1e-4
num_epochs = 50


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),  # ImageNet standards
                         (0.229, 0.224, 0.225))
])

########################################
# Training and Evaluation
########################################
def train_and_evaluate():
    # Paths (Update these paths as per your directory structure)
    root_folder = 'Flicker8k/Images'  # Path to images
    annotation_file = 'Flicker8k/Flickr8k.token.txt'  # Path to captions

    # Create DataLoaders
    full_dataset = Flickr8kDataset(
        root_dir=root_folder,
        captions_file=annotation_file,
        transform=transform
    )

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))  # Seed for reproducibility

    print(f"Dataset split into {train_size} training, {val_size} validation, and {test_size} test samples.")

    pad_token_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=MyCollate(pad_token_id=pad_token_id)
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=MyCollate(pad_token_id=pad_token_id)
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=MyCollate(pad_token_id=pad_token_id)
    )

    # Initialize models
    vocab_size = tokenizer.vocab_size
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Lists to store losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        running_train_loss = 0.0
        for idx, batch in enumerate(train_loader):
            # Skip if batch is None (all broken samples)
            if batch is None:
                continue

            images, captions, img_ids = batch
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            # Forward pass
            features = encoder(images)  # (B, 49, embed_size)
            outputs = decoder(features, captions[:-1])  # (T-1, B, vocab_size)

            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:].reshape(-1))
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Average train loss for the epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        encoder.eval()
        decoder.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if batch is None:
                    continue
                images, captions, img_ids = batch
                images = images.to(device)
                captions = captions.to(device)

                features = encoder(images)
                outputs = decoder(features, captions[:-1])

                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:].reshape(-1))
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the trained models
    torch.save(encoder.state_dict(), 'encoder_resnet101_attention-test.ckpt')
    torch.save(decoder.state_dict(), 'decoder_resnet101_attention-test.ckpt')
    print("Models saved successfully.")

    # Plot the learning curves
    plt.figure(figsize=(10,5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    plt.show()

    # After training, evaluate on the test set
    evaluate_on_test_set(encoder, decoder, test_loader, device)

def evaluate_on_test_set(encoder, decoder, test_loader, device):
    encoder.eval()
    decoder.eval()
    print("Evaluating on Test Set...")

    # Dictionary to hold image IDs and their captions
    test_img_captions = {}

    # Collect all ground truth captions
    for batch in test_loader:
        if batch is None:
            continue
        images, captions, img_ids = batch
        for img_id, cap in zip(img_ids, captions):
            if img_id not in test_img_captions:
                test_img_captions[img_id] = []
            # Decode caption tensor back to string
            cap_str = tokenizer.decode(cap, skip_special_tokens=True)
            # Tokenize the caption
            cap_tokens = nltk.word_tokenize(cap_str.lower())
            test_img_captions[img_id].append(cap_tokens)


    unique_img_ids = list(test_img_captions.keys())
    total_images = len(unique_img_ids)
    print(f"Total unique images in test set: {total_images}")

    bleu_scores = []
    meteor_scores = []

    for idx, img_id in enumerate(unique_img_ids):
        img_path = os.path.join('Flicker8k/Images', img_id)  # Update path if different
        try:
            image = Image.open(img_path).convert("RGB")
        except (IOError, UnidentifiedImageError) as e:
            print(f"Cannot open image {img_path}: {e}")
            continue

        # Preprocess the image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),  # ImageNet standards
                                 (0.229, 0.224, 0.225))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = encoder(image_tensor)  # (1, 49, embed_size)
            # Generate caption using beam search
            predicted_ids = decoder.beam_search(features, beam_width=3, max_len=20)
            # Convert to text
            predicted_caption = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            # Tokenize the generated caption
            hypothesis = nltk.word_tokenize(predicted_caption.lower())

        # Get ground truth captions
        references = test_img_captions[img_id]  # List of token lists

        # Compute BLEU score (using BLEU-4)
        bleu_score_val = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu_score_val)

        # Compute METEOR score
        # Pass lists of tokens directly without joining into strings
        meteor_score_val = meteor_score(references, hypothesis)
        meteor_scores.append(meteor_score_val)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            print(f"Processed {idx+1}/{total_images} images.")

    # Calculate average BLEU and METEOR scores
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
    print(f"Average BLEU-4 Score on Test Set: {avg_bleu:.4f}")
    print(f"Average METEOR Score on Test Set: {avg_meteor:.4f}")

########################################
# Testing Functionality
########################################
def test_model_on_new_images():
    """
    Allows uploading new images and generates captions using the trained model.
    If ground truth captions are available, it also computes BLEU and METEOR scores.
    """
    import tkinter as tk
    from tkinter import filedialog

    # Load the trained models
    embed_size = 512
    hidden_size = 512
    num_layers = 1
    vocab_size = tokenizer.vocab_size

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Load model weights
    encoder_checkpoint = 'encoder_resnet101_attention-test.ckpt'
    decoder_checkpoint = 'decoder_resnet101_attention-test.ckpt'

    if os.path.exists(encoder_checkpoint) and os.path.exists(decoder_checkpoint):
        encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=device))
        decoder.load_state_dict(torch.load(decoder_checkpoint, map_location=device))
        print("Trained models loaded successfully.")
    else:
        print("Model checkpoint files not found. Please ensure the paths are correct.")
        return

    encoder.eval()
    decoder.eval()

    # Function to generate caption for a single image
    def generate_caption_for_image(image_path, use_beam_search=True, beam_width=3):
        try:
            image = Image.open(image_path).convert("RGB")
        except (IOError, UnidentifiedImageError) as e:
            print(f"Cannot open image {image_path}: {e}")
            return None

        # Preprocess the image
        transform_single = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),  # ImageNet standards
                                 (0.229, 0.224, 0.225))
        ])
        image_tensor = transform_single(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = encoder(image_tensor)  # (1, 49, embed_size)
            if use_beam_search:
                predicted_ids = decoder.beam_search(features, beam_width=beam_width, max_len=20)
            else:
                predicted_ids = decoder.sample(features, max_len=20)
            predicted_caption = tokenizer.decode(predicted_ids, skip_special_tokens=True)

        return predicted_caption

    # Initialize Tkinter root
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select images
    file_paths = filedialog.askopenfilenames(title='Select Images for Captioning',
                                             filetypes=[('Image Files', '*.jpg;*.jpeg;*.png')])
    if not file_paths:
        print("No images selected.")
        return

    for img_path in file_paths:
        print(f"\nProcessing Image: {img_path}")
        caption = generate_caption_for_image(img_path, use_beam_search=True, beam_width=3)
        if caption:
            print(f"Generated Caption: {caption}")


            # Here, we'll skip BLEU and METEOR for user-uploaded images unless ground truths are provided
            # You can extend this part based on how you store ground truths for new images
            # Example:
            # ground_truths = get_ground_truth_captions(img_path)
            # if ground_truths:
            #     references_tokenized = [nltk.word_tokenize(ref.lower()) for ref in ground_truths]
            #     hypothesis_tokenized = nltk.word_tokenize(caption.lower())
            #     bleu = sentence_bleu(references_tokenized, hypothesis_tokenized)
            #     meteor = meteor_score([' '.join(ref) for ref in references_tokenized], ' '.join(hypothesis_tokenized))
            #     print(f"BLEU Score: {bleu:.4f}, METEOR Score: {meteor:.4f}")
        else:
            print("Failed to generate caption.")

########################################
# Main Execution
########################################
if __name__ == '__main__':
    # Train and evaluate the model
    train_and_evaluate()

    # After training, test the model on new images
    # Uncomment the following line to enable testing on new images
    # test_model_on_new_images()
