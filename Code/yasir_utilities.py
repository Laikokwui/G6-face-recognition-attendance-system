import io
import os

IMAGE_SIZE = (64,64)
DATABASE_DIR = r"C:\Users\HP\Desktop\GitHub\G6-face-recognition-attendance-system\Code\database"

_image_database = {}

def _remove_extension(fileName:str) -> str:
    index = fileName.rfind(".")
    return fileName[0:index]

def _get_image_embedding(kerasModel:Keras.Model, fileName:str) -> NumPy.ndarray:
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
    embeddingDifference = embedding1 - embedding2
    return math.tanh(NumPy.square(embeddingDifference).sum())

def _load_image_database(smlb:SimpleMLBuilder) -> dict:
    kerasModel = smlb.get_compiled_model()
    database = {}
    for currentDirectory, directoryNames, fileNames in os.walk(DATABASE_DIR):
        for fileName in fileNames:
            database[_remove_extension(fileName)] = _get_image_embedding(kerasModel, fileName)
    return database

def _get_image_database(smlb:SimpleMLBuilder) -> dict:
    if _image_database.empty():
        _image_database = _load_image_database(smlb)
    
    return _image_database

def add_image_to_database(imageJpegFilePath:str):
    with open(imageJpegFilePath, "rb") as imageBinaryRead:
        newFileNameStart = imageJpegFilePath.rfind(os.path.sep)+1
        newFileName = imageJpegFilePath[newFileNameStart:]
        with open(DATABASE_DIR + os.path.sep + newFileName, "wb") as imageBinaryWrite:
            imageBinaryWrite.write(imageBinaryRead.read())

def get_closest_image(imageJpegFilePath:str, threshold:float) -> str:
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
    
    kerasModel = smlb.get_compiled_model()
    compEmbedding = kerasModel(imageRawResized)
    bestName = ""
    bestDistance = 1
    for name, embedding in _get_image_database(smlb):
        embeddingDistance = get_embedding_distance(embedding, compEmbedding)
        if embeddingDistance < bestDistance:
            bestName = name
            bestDistance = embeddingDistance
    
    if bestDistance < thresholdDistance:
        return bestName
    else:
        return ""