import io
import math
import os

import sys, math

def smlb_log(*message, sep:str=" "):
    print("[SMLB]", *message, sep=sep)

def smlb_log_error(*message, sep:str=" "):
    print("[SMLB]", *message, sep=sep, file=sys.stderr)

smlb_log("Importing modules...")

smlb_log("Loading TensorFlow... this may take a long while.")
import tensorflow as TensorFlow
Keras = TensorFlow.keras
smlb_log("TensorFlow loaded! TensorFlow version is", TensorFlow.__version__ + ".")

smlb_log("Loading NumPy...")
import numpy as NumPy
smlb_log("NumPy loaded!")

smlb_log("Loading PyPlot...")
from matplotlib import pyplot as PyPlot
smlb_log("PyPlot loaded!")

smlb_log("All imports successful!")

class SimpleMLBuilder:
    def __init__(self, verbose:bool=False):
        self.verbose = verbose
        self.log("Fully initialized!")

    def log(self, *message, sep:str=" ", nonVerbose:bool=False):
        if self.verbose or nonVerbose:
            smlb_log(*message, sep=sep)

    def get_compiled_model(self) -> Keras.Model:
        return self.compiledModel

    def load(self, name:str):
        self.log("Loading model...")
        self.compiledModel = Keras.models.load_model(name)
        self.log("Load complete!")

smlb_log("Initialization successful!")



IMAGE_SIZE = (64,64)
DATABASE_DIR = r"C:\Users\HP\Desktop\GitHub\G6-face-recognition-attendance-system\Code\database"

_image_database = {}

# All functions with an underscore in front of their name are INTERNAL FUNCTIONS. Do not use them!

def _remove_extension(fileName:str) -> str:
    index = fileName.rfind(".")
    return fileName[0:index]

def _get_image_embedding(kerasModel:Keras.Model, absolutePath:str) -> NumPy.ndarray:
    # Load the image via TensorFlow.io.read_file
    imageBinary = TensorFlow.io.read_file(absolutePath)
    # Decode the binary via TensorFlow.io.decode_jpeg
    imageRaw = TensorFlow.io.decode_jpeg(imageBinary, channels=3)
    # Resize the raw image in case that the image is not 64x64
    imageRawResized = TensorFlow.image.resize(
        imageRaw, IMAGE_SIZE, method=TensorFlow.image.ResizeMethod.BICUBIC
    )
    imageRawReshaped = TensorFlow.reshape(imageRawResized, (1, 64, 64, 3))
    return kerasModel(imageRawReshaped)

def _get_embedding_distance(embedding1, embedding2) -> float:
    """Returns the distance between two embeddings."""
    embeddingDifference = embedding1 - embedding2
    return NumPy.square(embeddingDifference).sum()

def _create_image_database(smlb:SimpleMLBuilder) -> dict:
    """Creates a database of image embeddings."""
    kerasModel = smlb.get_compiled_model()
    database = {}
    for currentDirectory, directoryNames, fileNames in os.walk(DATABASE_DIR):
        for fileName in fileNames:
            database[_remove_extension(fileName)] = _get_image_embedding(kerasModel, os.path.join(currentDirectory, fileName))
    return database

def _get_image_database(smlb:SimpleMLBuilder) -> dict:
    nonlocal _image_database
    if _image_database.empty():
        _image_database = _create_image_database(smlb)

    return _image_database

def add_image_to_database(imageJpegFilePath:str):
    """Adds an image to the database.

    imageJpegFilePath should be a string describing the path to the file (e.g. "C:/Users/User/Desktop/Henry.jpg").
    Note that the file name excluding the extension ("Henery" in the above example) will be used as the database key for the embedding."""
    with open(imageJpegFilePath, "rb") as imageBinaryRead:
        newFileNameStart = imageJpegFilePath.rfind(os.path.sep)+1
        newFileName = imageJpegFilePath[newFileNameStart:]
        with open(os.path.join(DATABASE_DIR, newFileName), "wb") as imageBinaryWrite:
            imageBinaryWrite.write(imageBinaryRead.read())

def get_closest_image(imageJpegFilePath:str, threshold:float) -> str:
    """Gets the database key for the closest image embedding, or "" if the distance is over the threshold.

    The database stores a dictionary / associative list structure of key-value pairs.
    The key will be the filename minus the file extension (e.g. "C:/Users/User/Desktop/Database/Henry.jpg" becomes "Henry").
    The value of the database is the embedding of the image.

    imageJpegFilePath should be a string describing the path to the file (e.g. "C:/Users/User/Desktop/Henry.jpg").
    threshold should be a float from 0-1. The lower the value, the more strict the system the comparison algorithm will be.
    If the distance does not pass the threshold value, an empty string will be returned.
    """
    smlb = SimpleMLBuilder()
    smlb.load("Final Model")

    # Load the image file
    imageBinary = TensorFlow.io.read_file(imageJpegFilePath)
    # Decode the image file
    imageRaw = TensorFlow.io.decode_jpeg(imageBinary, channels=3)
    # Resize the raw image
    imageRawResized = TensorFlow.image.resize(
        imageRaw, IMAGE_SIZE, method=TensorFlow.image.ResizeMethod.BICUBIC
    )
    imageRawReshaped = TensorFlow.reshape(imageRawResized, (1, 64, 64, 3))

    kerasModel = smlb.get_compiled_model()
    compEmbedding = kerasModel(imageRawReshaped)
    bestName = ""
    bestDistance = math.inf
    worstDistance = 0
    for name, embedding in _get_image_database(smlb):
        embeddingDistance = get_embedding_distance(embedding, compEmbedding)
        if embeddingDistance < bestDistance:
            bestName = name
            bestDistance = embeddingDistance
        if embeddingDistance > worstDistance:
            worstDistance = embeddingDistance

    bestDistance /= worstDistance

    if bestDistance < threshold:
        return bestName
    else:
        return ""