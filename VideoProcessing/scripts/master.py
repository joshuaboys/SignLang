import csv
import glob
import os
import os.path
import numpy as np
import time
import sys
import ffmpeg
from subprocess import call, Popen
from models import ResearchModels
from data import DataSet
from extractor import Extractor
from tqdm import tqdm
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

# folder which includes the videos
data_path = sys.argv[1]
# folder for model output
model_path = sys.argv[2]

def extract_files():
    """We need to make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    videos = glob.glob(os.path.join(data_path,'*.mp4'))
	
    for video_path in videos:
        # Get the parts of the file.
        video_parts = get_video_parts(video_path)
        classname, filename_no_ext, filename = video_parts
        print("Video Parts:" + model_path + "|"+ classname +"|"+ filename_no_ext +"|"+ filename)
		
        #create class dir if not exists
        if not os.path.exists(os.path.join(model_path, classname)):
            os.makedirs(os.path.join(model_path, classname))
			
        # Only extract if we haven't done it yet. Otherwise, just get
        # the info.
        if not check_already_extracted(video_parts):
            dest = os.path.join(model_path, classname, filename_no_ext + '-%04d.jpg')

            print("extract:" + video_path + " | " + " dest: " + dest)
			#Split video into frames
            ffmpeg.input(video_path).output(dest).run()
            #os.system("wine" + model_path+'/ffmpeg.exe -i ' + video_path + ' ' + dest)

        # Now get how many frames it is.
        nb_frames = get_nb_frames_for_video(video_parts)

        data_file.append([classname, filename_no_ext, nb_frames])

        print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

		
    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)
		
				
    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join(classname, filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""

    parts = video_path.split("/")[-1].split('_')
    classname = parts[0]
    filename = parts[1]
    filename_no_ext = filename.split('.')[0]

    return classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join(model_path, classname, filename_no_ext + '-0001.jpg')))

def extract_features():
    # Set defaults.
    seq_length = 30
    class_limit =  None  # Number of classes to extract. Can be 1-101 or None for all.
    
    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit)
    
    # get the model.
    model = Extractor()
    print(data.data)
    
    # Loop through data.
    pbar = tqdm(total=len(data.data))
    for video in data.data:
    
    	# Get the path to the sequence for this video.
    	path = os.path.join(model_path, 'data','sequences', video[1] + '-' + str(seq_length) + '-features')  # numpy will auto-append .npy
    
    	# Check if we already have it.
    	if os.path.isfile(path + '.npy'):
    		pbar.update(1)
    		continue
    
    	# Get the frames for this video.
    	frames = data.get_frames_for_sample(video)
    
    	# Now downsample to just the ones we need.
    	frames = data.rescale_list(frames, seq_length)
    
    	# Now loop through and extract features to build the sequence.
    	sequence = []
    	for image in frames:
    		features = model.extract(image)
    		sequence.append(features)
    
    	# Save the sequence.
    	np.save(path, sequence)
    
    	pbar.update(1)
    
    pbar.close()

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=10, nb_epoch=100): #batch_size=32
    _model_path = os.path.join(model_path, 'data', 'checkpoints')
    #create class dir if not exists
    if not os.path.exists(_model_path):
            os.makedirs(_model_path)
			
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(_model_path, model + '-' + data_type + '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join(model_path,'data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=50)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(model_path,'data', 'logs', model + '-' + 'training-' + str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'test', data_type)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[tb, early_stopper, csv_logger, checkpointer],
            validation_data=val_generator,
            validation_steps=40,
            workers=4)

def train_model():
    """These are the main training settings. Set each before running
    this file."""
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit =  None  # int, can be 1-101 or None
    seq_length = 30
    load_to_memory = False  # pre-load the sequences into memory
    batch_size = 5
    nb_epoch = 1000

    # Chose images or features and image shape based on network.
    if model in ['conv_3d', 'c3d', 'lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

extract_files()
extract_features()
train_model()
