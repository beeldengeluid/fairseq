


import shap
import torch
import numpy as np
from PIL import Image
import os, copy, sys
import math, json
import random
from tqdm import tqdm
import fairseq
from fairseq import checkpoint_utils, options, progress_bar, utils
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
import matplotlib.colors as mcolors

## This file contains the explainability framework merged with the model 'sse-ft', it is a work in progress, however the masking logic is tested and is correct

## Main for now parses the args from the command prompt and passes them to load_models()


# Load_models() gets the model from the fairseq library and sets the checkpoint, it initialises a data iterator and loops over the samples in the dataset. 
# For now I need to test the masking functions together with the model, later i will adjust the code so that it can handle data from Sound and Vision
global text_tensor
global audio_tensor
global video_tensor
global text_length
global audio_length
global padded_amount_value
global samples_per_token
global task
global model
global criterion
global use_cuda
global target
global prediction
global video_depth
global label
global vocab


# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the desired log level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # log to console
        RotatingFileHandler('output.log', maxBytes=1024*1024, backupCount=5)  # save to file
    ]
)
logger = logging.getLogger(__name__)



def load_models(args, override_args=None):
    global task
    global model
    global criterion
    global use_cuda
    global label
    utils.import_user_module(args)
    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu
    
    if override_args is not None:
            overrides = vars(override_args)
            overrides.update(eval(getattr(override_args, 'model_overrides', '{}')))
    else:
        overrides = None
    ## Load ensemble and extract model
    logger.info('| loading model(s) from {}'.format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        )
    model = models[0]

        ## Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()
    logger.info(model_args)

    ## Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    ## Load valid dataset 
    for subset in args.valid_subset.split(','):
        try:
            task.load_dataset(subset, combine=False, epoch=0)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception('Cannot find dataset: ' + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        # progress = progress_bar.build_progress_bar( outcommented
        #     args, itr,
        #     prefix='valid on \'{}\' subset'.format(subset),
        #     no_progress_bar='simple'
        # )

    labels = ['Neutral','Sadness', 'Anger', 'Joy', 'Surprise', 'Fear', 'Disgust']
    #label = ['Neutral']   
    # Create a dictionary mapping token IDs to words
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    token_to_word = {i: tokenizer.decode(i, skip_special_tokens=True) for i in range(tokenizer.vocab_size)}
    # Define the token IDs for Friends characters

    token_to_word.update({7423: "Rachel"})
    token_to_word.update({4129: "Phoebe"})
    token_to_word.update({3540: "Phoebe"})
    token_to_word.update({1610: "Phoebe"})
    token_to_word.update({4012: "Ross"})
    token_to_word.update({12811: "Monica"})
    token_to_word.update({16027: "Chandler"})
    token_to_word.update({12972: "Joey"})

    results_dict = {} ## This dict will gatter the shap values for each data sample key: sample ID value: shap-values for each combination of tokens
    total_text_score = 0
    total_audio_score = 0
    total_video_score = 0
    num_samples = len(results_dict)
    ## This loop iterates over every sample and creates the tokens representing the data input for each modality (shap needs tokens instead if the real input data)
    # for index, sample in enumerate(progress):
    for index, sample in enumerate(itr):
        #DATA TEST, the code below is made for testing
        #x = {'net_input': {'audio': audio_tensor, 'text': text_tensor, 'video': video_tensor, 'padded_amount':padded_amount_value }}
        #audio_tensor = torch.randn(1, 150000)  # Example dimensions (batch_size, channels, audio_length)
        #text_tensor = torch.randn(1, 30)  # Example dimensions (batch_size, seq_length, embedding_size)
        #video_tensor = torch.randn(1, 3, 300, 256, 256)  # Example dimensions (batch_size, channels, frames, height, width)
        #padded_amount_value = 100
        results_dict[index] = {}
        for label in labels:
            global audio_tensor
            global text_tensor
            global video_tensor
            global padded_amount_value
            global text_length
            global audio_length
            global samples_per_token
            global video_length
            global target
            global prediction
            global video_depth
            audio_tensor = sample['net_input']['audio']
            text_tensor = sample['net_input']['text']
            video_tensor = sample['net_input']['video']
            padded_amount_value = sample['net_input']['padded_amount']
            target = sample['target']
            ## Text tokens
            text = text_tensor.squeeze(0).cpu().numpy()
            #last_non_zero_index = np.max(np.nonzero(text))
            #text_length  = len(text[:last_non_zero_index + 1])
            text = text[text != 1]
            text_length = len(text)
            text_tokens_ids = torch.tensor(range(1, text_length+1)).unsqueeze(0)
        
            ## Audio tokens, each token represents a segment of a second in spectogram values (the sanmple rate is used to calculate the amount of values per token)
            audio = audio_tensor.squeeze(0).cpu().numpy()
            last_index = len(audio) - np.argmax(np.flip(audio) != 0) - 1
            audio = audio[:last_index + 1]
            sample_rate = 16000
            sample_duration = 1.0 / sample_rate
            token_duration = 1.0
            samples_per_token = int(token_duration / sample_duration)
            audio_length = math.ceil((len(audio)/samples_per_token))
            audio_tokens_ids = torch.tensor(range(1, audio_length+1)).unsqueeze(0) 
    
            ## Video tokens, each pixel is represented as a token (256 x 256 is an image) 
            video_token_ids = torch.tensor(range(1, 17)).unsqueeze(0)
            #video_token_ids = torch.tensor(range(1, 257)).unsqueeze(0) (for larger pixels)
            video_depth = find_length_video(video_tensor)
            video_tokens = np.ceil(video_depth / audio_length).astype(int)
            temp_video_token_ids = torch.tensor(range(1, video_tokens+1))
            ## All tokens are concatenated together
            All_token_ids = torch.cat((text_tokens_ids, audio_tokens_ids, video_token_ids), 1).unsqueeze(1)
            ## A SHAP explainer object is made with the custom masker and the model prediction function
            explainer = shap.Explainer(
            get_model_prediction, custom_masker, silent=True, output_names=labels)
            shap_values = explainer(All_token_ids)
            logger.info(shap_values)
            ## Shap values are stored
            text_score, audio_score, video_score = compute_mm_score_local(shap_values)
            logger.info("Label: %s, TEXT_score: %s, AUDIO_score: %s, VIDEO_score: %s", label, text_score, audio_score, video_score)
            #results_dict[index][f'shap_values_{label}'] = shap_values outcomment
            total_text_score += text_score
            total_audio_score += audio_score
            total_video_score += video_score
            image = video_tensor[:, 0, 1, :, :].squeeze().numpy()
            shap_image = shap_values.values[0,0,text_length+audio_length:]
            shap_text = shap_values.values[0,0,:text_length]
            shap_audio = shap_values.values[0,0,text_length:text_length+audio_length]
            visualize_image_shap(shap_image, image, label) # visualize shap values for each image represented by tokens
            visualize_text_shap(shap_text, token_to_word, text, label)
            visualize_audio_shap(shap_audio,audio,audio_length, label)
            visualize_audio_text_shap(shap_text, shap_audio, text, token_to_word, label)
        max_modality = np.max([text_score, audio_score, video_score])
        missclassified = prediction != target
        results_dict[index].update = ({
            'prediction': prediction,
            'target': target,
            'max_modality': max_modality,
            'missclassified': missclassified,
            'text_length': text_length,
            'audio_length': audio_length})
        total_text_score = total_text_score / 7 #Normalize for the all labels
        total_audio_score = total_audio_score / 7
        total_video_score = total_video_score / 7
        logger.info("The predicted emotion is:" , prediction)  
        logger.info("Over all labels: TEXT_score: %s, AUDIO_score: %s, VIDEO_score: %s", text_score, audio_score, video_score)

        
    avg_text_score = total_text_score / num_samples # Normalize for all samples
    avg_audio_score = total_audio_score / num_samples
    avg_video_score = total_video_score / num_samples
    # Print the average multimodality scores for the entire dataset for one label
    logger.info("Average Text Score: %s", avg_text_score)
    logger.info("Average Audio Score: %s", avg_audio_score)
    logger.info("Average Video Score: %s", avg_video_score)
    
## The custom masker recieves the input tokens and a boolean tensor of the same shape (what need to be masked)
def custom_masker(mask, x):
    masked_X = x.clone()
    mask = torch.tensor(mask).unsqueeze(0)
    masked_X[~mask] = 0  # ~mask !!! to zero
    # never mask out CLS and SEP tokens (makes no sense for the model to work without them) -> this is handled in the load_models range(1,...) instead of range*(0,....)
    return masked_X

## This function recieves an array with rows that represent combinations of masked tokens for one sample
## From the masked tokens, based on the token generaton logic, the input is reconstructed to the form needed by the model
def get_model_prediction(x):
    global prediction
    with torch.no_grad():

        masked_text_tokens_ids = torch.tensor(x[:, :text_length]) 
        masked_audio_tokens_ids = torch.tensor(x[:, text_length: text_length + audio_length])
        masked_video_tokens_ids = torch.tensor(x[:, text_length + audio_length:])
        result = np.zeros(x.shape[0]) 
        for i in range(x.shape[0]):
            masked_text_tensor = text_tensor.clone()
            for k in range(1, len(masked_text_tokens_ids[0])):
                if masked_text_tokens_ids[i,k].item() == 0:
                    masked_text_tensor[0,k] = 0 ## Here the real text masking happens
            masked_audio_tensor = audio_tensor.clone()
            for k in range(masked_audio_tokens_ids.size(1)):
                if masked_audio_tokens_ids[i,k].item() == 0:
                    masked_audio_tensor[k:k+samples_per_token] = 0 ## Here the real audio masking happens
            masked_video_tensor = video_tensor.clone()
            #num_patches = 256
            num_patches = 16
            patch_size = 64
            num_frames = 300
            num_patches_per_row = 4
            #Patches as tokens
            for k in range(masked_video_tokens_ids.size(1)-1):
                if masked_video_tokens_ids[i,k].item() == 0 :
                    # start_row = (k // (256 // patch_size)) * patch_size
                    # start_col = (k % (256 // patch_size)) * patch_size
                    start_row = (k // num_patches_per_row) * patch_size 
                    start_col = (k % num_patches_per_row) * patch_size
                    for frame_idx in range(masked_video_tensor.shape[2]):
                      if torch.any(masked_video_tensor[:,:,frame_idx,:,:]== -1):
                        continue  # Skip padded frames
                      masked_video_tensor[:,:, frame_idx, start_row:start_row+patch_size, start_col:start_col+patch_size] = 0 ## Here the real video masking happens
            frames_per_token = video_depth // masked_video_tokens_ids.size(1)
            #Timeframes as tokens
            # for k in range((masked_video_tokens_ids.size(1)-1)):
           #     start_frame = k * frames_per_token
            #    end_frame = min((k + 1) * frames_per_token, video_depth)
            #     if masked_video_tokens_ids[i,k].item() == 0 :
            #         masked_video_tensor[:,:,start_frame:end_frame,:,:] = 0
            ## Input in the right shape fir model
            id_combination = torch.tensor([0])
            target = torch.tensor([0.])

            #masked_input = {'net_input': {'audio': masked_audio_tensor, 'text': masked_text_tensor, 'video': masked_video_tensor, 'padded_amount':padded_amount_value }}
            masked_model_input = {'id': id_combination, 'net_input': {'audio': masked_audio_tensor, 'text': masked_text_tensor, 'video': masked_video_tensor, 'padded_amount':padded_amount_value}, 'target': target} #TODO change id
            masked_model_input = utils.move_to_cuda(masked_model_input) if use_cuda else masked_model_input
            ## Collect the prediction andthe probabilities, let the model infer
            prediction, probabilities = task.valid_step(masked_model_input, model, criterion)
            ## Store the outcome of the model
            probabilities = np.array(probabilities)
            probability_class_max = np.max(probabilities)
            probability_neut = probabilities[0]
            if label == "Neutral":
                probability_class = probabilities[0]
            if label == "Sadness":
                probability_class = probabilities[1]
            if label == "Anger":
                probability_class = probabilities[2]
            if label == "Joy":
                probability_class = probabilities[3]
            if label == "Surprise":
                probability_class = probabilities[4]
            if label == "Fear":
                probability_class = probabilities[5]
            if label == "Disgust":
                probability_class = probabilities[6]
            result[i] = probability_class
    return result
#if torch.any(masked_video_tensor[0, :, frame_idx] == -1): TEST
def compute_mm_score_local(shap_values):
    """ Compute Multimodality Score. . """
    text_contrib = np.abs(shap_values.values[0, 0, :text_length]).sum()/text_length 
    audio_contrib = np.abs(shap_values.values[0, 0, text_length: text_length + audio_length]).sum()/audio_length
    video_contrib = np.abs(shap_values.values[0, 0, text_length + audio_length:]).sum()/16
    # text_contrib = np.abs(shap_values.values[:, :, :text_length]).sum(axis=(0, 1)) / text_length
    # audio_contrib = np.abs(shap_values.values[:, :, text_length: text_length + audio_length]).sum(axis=(0, 1))/audio_length
    # video_contrib = np.abs(shap_values.values[:, :, text_length + audio_length:]).sum(axis=(0,1))/16
    text_score = text_contrib / (text_contrib + video_contrib + audio_contrib)
    audio_score = audio_contrib / (text_contrib + video_contrib + audio_contrib)
    video_score = video_contrib / (text_contrib + video_contrib + audio_contrib)
    # image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
    return text_score, audio_score, video_score

def find_length_video(frames):
    time_depth = frames.shape[2]
    for t in range(time_depth):
        if torch.any(frames[:,:, t, :, :] == -1):
            return t #The start of the padding 
    return time_depth

def visualize_text_shap(shap_values, token_to_word, tokenized_text, label):
    # Sort tokens based on SHAP values
    print(shap_values)
    sorted_indices = np.argsort(shap_values)[::-1]  # Sort in descending order
    print(sorted_indices)
    valid_indices = [i for i in sorted_indices if i < len(tokenized_text)] # Check for index faults
    sorted_tokens = [token_to_word[tokenized_text[i]] for i in valid_indices]
    print(sorted_tokens)
    colors = ['red' if shap_values[i] >= 0 else 'blue' for i in valid_indices] # Color mapping
    plt.figure()
    # Plot the words in order of importance with colors
    #plt.figure(figsize=(10, 5))
    plt.barh(range(len(valid_indices)), shap_values[valid_indices], color=colors)
    plt.yticks(range(len(valid_indices)), sorted_tokens)
    plt.xlabel('SHAP Value')
    plt.ylabel('Word')
    plt.title(f'Words in Order of Importance based on SHAP Values for label {label}')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important word at the top
    plt.savefig(f"Text_plot-{label}.png")

def visualize_image_shap(shap_values, image, label):
    print(image)
    print(shap_values)
    shap_overlay = np.zeros_like(image)
    token_grid_size = 4
    token_size = len(image) // token_grid_size

    for i, shap_value in enumerate(shap_values):
        row_idx = i // token_grid_size
        col_idx = i % token_grid_size
        start_row = row_idx * token_size
        end_row = (row_idx + 1) * token_size
        start_col = col_idx * token_size
        end_col = (col_idx + 1) * token_size
        shap_overlay[start_row:end_row, start_col:end_col] = shap_value
    #plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Originele Afbeelding')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    plt.imshow(shap_overlay, cmap='coolwarm', alpha=0.5)  
    plt.title(f'SHAP Overlay - {label}')
    plt.axis('off')
    plt.savefig(f"Image_plot-{label}.png")

def visualize_audio_shap(shap_values,audio_values,audio_length, label):
    audio_length = len(audio_values)
    shap_length = len(shap_values)
    print(shap_values)
    # Interpolate the SHAP values to match the length of the audio waveform
    shap_repeated = np.repeat(shap_values, audio_length // shap_length)
    total_time = np.arange(audio_length)
    plt.figure()

    # Plot audio waveform
    plt.subplot(2, 1, 1)
    plt.plot(total_time, audio_values, color='black', label='Audio waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    plt.legend()
    plt.grid(True)

    # Plot SHAP values as bars covering the entire audio waveform
    plt.subplot(2, 1, 2)
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', ['blue', 'white', 'red'])
    plt.bar(total_time, shap_repeated, width=1, color=cmap((shap_repeated - shap_repeated.min()) / (shap_repeated.max() - shap_repeated.min())))
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized SHAP Value')
    plt.title(f'Normalized SHAP Values over Audio Waveform - {label}')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"Audio_plot-{label}.png")

def visualize_audio_text_shap(shap_values_text, shap_values_audio, tokenized_text, token_to_word, label):

    # Get words from tokenized text using the token_to_word dictionary
    words = [token_to_word[token] for token in tokenized_text]

    # Plot text SHAP values
    plt.figure()
    bar_colors = ['red' if shap >= 0 else 'blue' for shap in shap_values_text]
    plt.bar(np.arange(len(words)), shap_values_text, color=bar_colors, alpha=0.5, label='Text SHAP Values')

    # Calculate the length of the sentence
    length_sentence = len(words)

    # Calculate the number of audio segments needed to match the number of text values
    num_audio_segments = len(shap_values_audio)

    # Calculate the number of words per audio segment
    words_per_segment = length_sentence // num_audio_segments

    # Repeat audio SHAP values for each word segment
    repeated_audio_shap = np.repeat(shap_values_audio, words_per_segment)

    # Plot repeated audio SHAP values as horizontal lines
    for i in range(num_audio_segments):
        segment_start = i * words_per_segment
        segment_end = min((i + 1) * words_per_segment, length_sentence)
        audio_values_segment = repeated_audio_shap[i * words_per_segment: (i + 1) * words_per_segment]
        line_color = 'red' if shap_values_audio[i] >= 0 else 'blue'  # Set color based on original audio SHAP value
        plt.plot(np.arange(segment_start, segment_end), audio_values_segment, color=line_color, linewidth=2)

    # Set x-axis ticks and labels
    plt.xticks(np.arange(len(words)), words)

    # Set labels and title
    plt.xlabel('Sentence')
    plt.ylabel('SHAP Values')
    plt.title(f'Comparison of Audio and Text SHAP Values-{label}')

    plt.plot([], [], color='red', linewidth=2, label='Audio SHAP Values')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"Audio_Text_Shap_compare-{label}.png")


def cli_main():

    # only override args that are explicitly given on the command line
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    load_models(args, override_args)
if __name__ == '__main__':
    cli_main()
