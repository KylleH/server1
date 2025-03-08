#add json folder for UwU monitor Data UwU , OwO temp rpm when OwO gets time
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException,Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import matplotlib.pyplot as plt
import time
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from scipy.signal import butter, filtfilt
from typing import Optional
from datetime import datetime

app = FastAPI()
from PIL import Image
from torchvision import transforms
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from PIL import Image


# Define data directories
# Update the model path to use the WSL path
model_path = "/mnt/c/Users/kylle/Desktop/Web app/venv/cnn_classifier.keras" #mnt for linux
model = load_model(model_path)
output_dir_pic = "output/pictures" # this is on server virtual stuff buddy
output_dir_data = "output/data"
output_dir_fft = "output/fft"
model_path = r"C:\Users\kylle\Desktop\Web app\venv\cnn_classifier.keras"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load the CNN model
model = load_model(model_path)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((854, 480))
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    file_path = os.path.join(output_dir_pic, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    image_array = preprocess_image(file_path)
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100  # Convert to percentage
    
    classes = ["adc noise", "vibration noise", "class_2", "class_3"]  # Modify with actual class names
    predicted_class = classes[class_index] if confidence >= 50 else "default_class"
    
    return {"status": "success", "class": predicted_class, "confidence": f"{confidence:.2f}%"}
            
# Load your trained model (adjust the path)
@app.post("/api/save_plot_to_dataset")
async def save_plot_to_dataset(category: str):
    try:
        # Define the save path
        save_dir = os.path.join("dataset", "test", category)
        os.makedirs(save_dir, exist_ok=True)

        # Copy the plot to the dataset folder
        plot_save_path = os.path.join(save_dir, f"plot_{int(time.time())}.png")
        os.rename(PLOT_DIR, plot_save_path)

        return {"status": "success", "message": f"Plot saved to {plot_save_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((854, 480)),  # Resize to match model input size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize (adjust based on your training)
])





# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the directories for saving the data
output_dir_pic = "acc_pic"
output_dir_data = "acc_data"
output_dir_fft = "acc_fft"
sampling_rate=1000


PLOT_DIR = "3phasepic" # pic of 3 phase
JSON_DIR = "3phase_time" #datalog of time
JSON_DIRf = "3phase_fft" #datalog of fft

#app.mount("/venv", StaticFiles(directory="venv"), name="venv")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="static")

# Serve the main HTML file
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



# Ensure the directories exist
os.makedirs(output_dir_pic, exist_ok=True)
os.makedirs(output_dir_data, exist_ok=True)
os.makedirs(output_dir_fft, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(JSON_DIRf, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
# Set the directory for the acc_data folder inside your virtual environment

# Create the folder if it doesn't exist
if not os.path.exists(output_dir_data):
    os.makedirs(output_dir_data)

# Store the most recent accelerometer data
latest_data = {}

class VibrationData(BaseModel):
    vibrationData: list

class ChannelData(BaseModel):
    channelData: list

class ADCData(BaseModel):
    recording_time: int
    data: list

class MonitorData(BaseModel):
    U_rms_current: Optional[float] = None
    U_rms_voltage: Optional[float] = None
    U_PF: Optional[float] = None
    V_rms_current: Optional[float] = None
    V_rms_voltage: Optional[float] = None
    V_PF: Optional[float] = None
    W_rms_current: Optional[float] = None
    W_rms_voltage: Optional[float] = None
    W_PF: Optional[float] = None
    Motor_Temperature: Optional[float] = None
    Ambient_Temperature: Optional[float] = None
    RPM:Optional[float]=None
    flag_1: Optional[bool] = False
    flag_2: Optional[bool] = False

# Folder to store JSON logs of monitor
json_folder = "statsjson"
if not os.path.exists(json_folder):
    os.makedirs(json_folder)

# Temporary storage for partial data
partial_data = {}

@app.post("/receive_data")
async def receive_data(data: MonitorData):
    global partial_data
    try:
        print("Received Data:", data.dict())

        # Update partial data with new values, but only update flags if they are True
        for key, value in data.dict().items():
            if value is not None:
                # For flags, only update if the new value is True
                if key in ["flag_1", "flag_2"]:
                    if value is True:
                        partial_data[key] = True
                else:
                    partial_data[key] = value

        # List of required fields to check for valid, non-zero values
        required_fields = [
            "U_rms_current", "U_rms_voltage", "U_PF",
            "V_rms_current", "V_rms_voltage", "V_PF",
            "W_rms_current", "W_rms_voltage", "W_PF",
            "Motor_Temperature", "Ambient_Temperature"
        ]

        # Check if all required fields are filled with valid, non-zero values
        all_fields_valid = all(
            partial_data.get(field) is not None and partial_data.get(field) != 0 and partial_data.get(field) != -273.15
            for field in required_fields
        )

        # Check if both flags are True
        flags_true = partial_data.get("flag_1", False) and partial_data.get("flag_2", False)

        # Save data only if all fields are valid and flags are True
        if all_fields_valid and flags_true:
            # Add a timestamp to the data
            partial_data["timestamp"] = datetime.utcnow().isoformat()

            # Save the combined data to a JSON file
            file_name = f"data_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
            file_path = os.path.join(json_folder, file_name)

            with open(file_path, 'w') as f:
                json.dump(partial_data, f, indent=4)

            # Reset only the flags
            partial_data["flag_1"] = False
            partial_data["flag_2"] = False

            return {"message": "New entry created", "file": file_name}

        return {"message": "Data received, waiting for more"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/receive_data")
async def get_data():
    try:
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
        if not json_files:
            raise HTTPException(status_code=404, detail="No data available")

        # Find the most recent file
        most_recent_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(json_folder, f)))
        file_path = os.path.join(json_folder, most_recent_file)

        # Read the most recent file
        with open(file_path, 'r') as f:
            data = json.load(f)

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def bandpass_filter(data, lowcut, highcut, fs, order=4):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            if low <= 0 or high >= 1:
                raise ValueError(f"Critical frequencies must satisfy 0 < low < high < Nyquist frequency. Got low={lowcut}, high={highcut}, Nyquist={nyquist}.")
            b, a = butter(order, [low, high], btype='band')
            return filtfilt(b, a, data)
        
def apply_window(data, lowcut=44, highcut=499):
    # Apply bandpass filter
            filtered_data = bandpass_filter(data, lowcut, highcut, 1000)
    # Apply Hanning window
            window = np.hanning(len(filtered_data))
            return filtered_data * window
    
@app.post("/api/adc") #adc handling
async def upload_data(adc_data: ADCData):
    recording_time = adc_data.recording_time
    data = adc_data.data

   # Calculate the sampling rate (assuming the recording time corresponds to the number of samples)
    sample_count = len(data)
    sampling_rate = sample_count / recording_time if recording_time > 0 else 0

    # Extract channels
    channels = list(zip(*data))  # Transpose rows to columns
    if len(channels) < 6:
        return {"error": "Insufficient channels in data"}

    # Plot data
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First 3 channels
    for i in range(3):
        axs[0].plot(channels[i], label=f"Channel {i+1}")
    axs[0].set_title("Channels 1-3")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("ADC Value")
    axs[0].legend()

    # Next 3 channels
    for i in range(3, 6):
        axs[1].plot(channels[i], label=f"Channel {i+1}")
    axs[1].set_title("Channels 4-6")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("ADC Value")
    axs[1].legend()
    # Generate timestamp
    timestamp = int(time.time())

# File paths with timestamp
    adc_PATH = os.path.join(PLOT_DIR, f"3phase_plot_{timestamp}.png")
    adcJSON_PATH = os.path.join(JSON_DIR, f"3phasedata_{timestamp}.json")
    fft_json_path = os.path.join(JSON_DIRf, f"fft_data{timestamp}.json")

 # Compute FFT for each channel
    fft_results = {}
    for i, channel in enumerate(channels):
        freq, magnitude = compute_fft(channel, sampling_rate * 1000)
        rounded_freq = np.round(freq).astype(int)  # Round frequencies to the nearest whole number
        fft_results[f"channel_{i+1}"] = {
        "frequency": rounded_freq.tolist(),
        "magnitude": magnitude.tolist()
    }

    # Save FFT results to JSON
    with open(fft_json_path, "w") as fft_json_file:
        json.dump(fft_results, fft_json_file, indent=4)

    # Plot data
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # First 3 channels
    for i in range(3):
        axs[0].plot(channels[i], label=f"Channel {i + 1}")
    axs[0].set_title("Channels 1-3")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("ADC Value")
    axs[0].legend()

    # Next 3 channels
    for i in range(3, 6):
        axs[1].plot(channels[i], label=f"Channel {i + 1}")
    axs[1].set_title("Channels 4-6")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("ADC Value")
    axs[1].legend()    

     # Save JSON data to file
    with open(adcJSON_PATH, "w") as json_file:
        json.dump({"recording_time": recording_time, "data": data}, json_file, indent=4)

    # Save the plot
    plt.tight_layout()
    plt.savefig(adc_PATH)
    plt.close()

    return {
        "message": "Data received and saved",
        "recording_time": recording_time,
        "plot_path": adc_PATH,
        "json_path": adcJSON_PATH,
    }

DATA_FOLDER = "acc_fft"

@app.get("/api/latest_data")
async def get_latest_data():
    try:
        # Get the list of all JSON files in the folder
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]

        # If no JSON files are found, return an error
        if not files:
            return {"status": "error", "message": "No data files found"}

        # Sort the files by creation time (most recent first)
        files.sort(key=lambda f: os.path.getctime(os.path.join(DATA_FOLDER, f)), reverse=True)

        # Get the most recent JSON file
        latest_file = files[0]

        # Construct the full path to the JSON file
        file_path = os.path.join(DATA_FOLDER, latest_file)

        # Read the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Return the JSON data
        return JSONResponse(content={"status": "success", "data": data})
    
    except Exception as e:
        return {"status": "error", "message": str(e)}
        
@app.get("/")
async def get_index():
    """
    Serve the HTML page for the frontend.
    """
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/api/latest_vibration")
async def get_latest_vibration_data():
    DATA_FOLDER = "acc_data"
    try:
        # Get the list of all JSON files in the folder
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]

        # If no JSON files are found, return an error
        if not files:
            return {"status": "error", "message": "No data files found"}

        # Sort the files by creation time (most recent first)
        files.sort(key=lambda f: os.path.getctime(os.path.join(DATA_FOLDER, f)), reverse=True)

        # Get the most recent JSON file
        latest_file = files[0]

        # Construct the full path to the JSON file
        file_path = os.path.join(DATA_FOLDER, latest_file)

        # Read the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Return the JSON data
        return JSONResponse(content={"status": "success", "data": data})
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/vibration")
async def receive_vibration_data(vibration_data: VibrationData):
    global ax_windowed, ay_windowed, az_windowed
    """
    Endpoint to receive accelerometer data and store the latest values.
    """
    global latest_data
    # Extract data for ax, ay, az from the incoming request
    ax_data = [entry["Ax"] for entry in vibration_data.vibrationData]
    ay_data = [entry["Ay"] for entry in vibration_data.vibrationData]
    az_data = [entry["Az"] for entry in vibration_data.vibrationData]

    # Update the latest data
    latest_data = {
        "ax_data": ax_data,
        "ay_data": ay_data,
        "az_data": az_data
    }
    

    ax_windowed = apply_window(ax_data)
    ay_windowed = apply_window(ay_data)
    az_windowed = apply_window(az_data)

  
    # Save the vibration data to a plot file (PNG)
    timestamp = int(time.time())
    plot_filename = os.path.join(output_dir_pic, f"vibration_analysis_{timestamp}.png")
    data_filename = os.path.join(output_dir_data, f"vibration_data_{timestamp}.json")
    
   
    try:
        # Plot the data and save it as a PNG
        plot_data_and_fft(ax_data, ay_data, az_data, plot_filename)

       

        # Save the vibration data (Ax, Ay, Az) as JSON
        vibration_data_dict = {
            "ax_data": ax_data,
            "ay_data": ay_data,
            "az_data": az_data
        }

        filtered_data_dict = {
        "ax_data": ax_windowed.tolist(),  # Convert ndarray to list
        "ay_data": ay_windowed.tolist(),  # Convert ndarray to list
        "az_data": az_windowed.tolist()   # Convert ndarray to list
            }

        with open(data_filename, "w") as f:
            json.dump(filtered_data_dict, f)
        print(f"Vibration data saved as '{data_filename}'")



      
    except Exception as e:
        return {"status": "error", "message": f"Error saving data: {str(e)}"}

    return {"status": "success", "message": "Data received, plot, and JSON saved."}

def compute_fft(data, sampling_rate):
            N = len(data)
            freq = np.fft.rfftfreq(N, 1 / sampling_rate)  # Frequency bins
            fft_values = np.fft.rfft(data)                # FFT calculation
            magnitude = np.abs(fft_values) / N           # Magnitude normalization
            return freq, magnitude
        

def plot_data_and_fft(ax_data, ay_data, az_data, plot_filename, sampling_rate=1000):
    """
    Plot the time-domain vibration data and the FFT for Ax, Ay, and Az,
    applying a Hanning window and focusing on frequencies within 6 Hz of 50 Hz.
    """
    try:
        
    
        # Apply window function to the data
       
        

        # Compute FFT for Ax, Ay, Az
        ax_freq, ax_fft = compute_fft(ax_windowed, sampling_rate)
        ay_freq, ay_fft = compute_fft(ay_windowed, sampling_rate)
        az_freq, az_fft = compute_fft(az_windowed, sampling_rate)

        # Filter frequencies between 50 Hz ± 6 Hz (44-56 Hz)
        def filter_frequencies(freq, fft_values, low_cut=44, high_cut=500):
            mask = (freq >= low_cut) & (freq <= high_cut)
            return freq[mask], fft_values[mask]

        ax_freq_filtered, ax_fft_filtered = filter_frequencies(ax_freq, ax_fft)
        ay_freq_filtered, ay_fft_filtered = filter_frequencies(ay_freq, ay_fft)
        az_freq_filtered, az_fft_filtered = filter_frequencies(az_freq, az_fft)

        fft_data_dict = {
            "ax_freq": ax_freq_filtered.tolist(),
            "ax_fft": ax_fft_filtered.tolist(),
            "ay_freq": ay_freq_filtered.tolist(),
            "ay_fft": ay_fft_filtered.tolist(),
            "az_freq": az_freq_filtered.tolist(),
            "az_fft": az_fft_filtered.tolist()
        }
        timestamp = int(time.time())
        fft_filename = os.path.join(output_dir_fft, f"fft_data_{timestamp}.json")
        with open(fft_filename, "w") as f:
            json.dump(fft_data_dict, f)
            

        # Plot time-domain data
        plt.figure(figsize=(12.8, 7.2), dpi=100)


        # Time-domain plot
        plt.subplot(2, 1, 1)
        plt.plot(ax_windowed, color='r')
        plt.plot(ay_windowed, color='g')
        plt.plot(az_windowed, color='b')
        plt.axis("off")
        plt.grid(False)
      

        # FFT plot (Filtered)
        plt.subplot(2, 1, 2)
        plt.plot(ax_freq_filtered, ax_fft_filtered,color='r')
        plt.plot(ay_freq_filtered, ay_fft_filtered,color='g')
        plt.plot(az_freq_filtered, az_fft_filtered,color='b')
        plt.axis("off")
        plt.grid(False)
    
        

        # Save the plot as a PNG image
        plt.savefig(plot_filename, dpi=100, bbox_inches='tight', pad_inches=0)
        print(f"Filtered vibration analysis saved as '{plot_filename}'")
        plt.close()

    except Exception as e:
        print(f"Error in plotting: {e}")

@app.post("/api/channel")
async def receive_channel_data(channel_data: ChannelData):
    """
    Endpoint to receive ADC channel data and log/display it.
    """
    data = channel_data.channelDataz
    print(f"Received channel data: {data}")

    # You can extend this to save the data, send it to a front-end, or process it further.
    return {"status": "success", "message": "Channel data received.", "data": data}


SCALE = float(os.getenv("SCALE", 1.0))  # Default to 1.0 if not found

@app.get("/api/adc1")
async def get_latest_data():
    try:
        # Get the list of all JSON files in the folder
        files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
        if not files:
            return {"status": "error", "message": "No data files found"}
        
        # Sort the files by creation time (most recent first)
        files.sort(key=lambda f: os.path.getctime(os.path.join(JSON_DIR, f)), reverse=True)
        latest_file = files[0]
        
        # Read the latest JSON data
        file_path = os.path.join(JSON_DIR, latest_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Scale the data values
        if "data" in data and isinstance(data["data"], list):
            data["data"] = [[(value - 2120) * SCALE if i < 3 else (value - 2150) * SCALE for i, value in enumerate(row)] for row in data["data"]]
        
        return JSONResponse(content={"status": "success", "data": data})
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/adc")
async def get_latest_fft():
    DATA_FOLDER = JSON_DIRf
    try:
        # Get the list of all JSON files in the folder
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]

        # If no JSON files are found, return an error
        if not files:
            return {"status": "error", "message": "No data files found"}

        # Sort the files by creation time (most recent first)
        files.sort(key=lambda f: os.path.getctime(os.path.join(DATA_FOLDER, f)), reverse=True)

        # Get the most recent JSON file
        latest_file = files[0]

        # Construct the full path to the JSON file
        file_path = os.path.join(DATA_FOLDER, latest_file)

        # Read the JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Return the JSON data
        return JSONResponse(content={"status": "success", "data": data})
    
    except Exception as e:
        return {"status": "error", "message": str(e)}

