let chartInstance = null; // Variable to hold the chart instance
let previousTimeData = null; // Declare previousTimeData globally
let intervalid = null; // Declare intervalId globally
const root = config.ROOT_URL; // Read from config.js

async function fetchLatestData() {
    try {
        const response = await fetch(`${root}8000/api/latest_vibration`); // Fixed string interpolation
        if (!response.ok) throw new Error("Network response was not ok");

        const result = await response.json();
        if (result.status === "success") {
            return result.data;
        } else {
            console.error("Error fetching data:", result.message);
            return null;
        }
    } catch (error) {
        console.error("Error fetching data:", error);
        return null;
    }
}

let previousTimeDataf = null; // Store the previous data for comparison

// Function to plot time-domain data
function plotData(labels, datasets, title) {
    const ctx = document.getElementById('timeDomainChart').getContext('2d');

    
    // Destroy existing chart instance if it exists
    if (chartInstance) {
        chartInstance.destroy();
    }

    // Create a new chart instance
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets // Use the datasets array for multiple data series
        },
        options: {
            responsive: true,
            animation: false, // Disable animation for faster updates
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Sample'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Amplitude'
                    }
                }
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", function () {
    const timebutton = document.getElementById("plot-time-domain");
    if (!timebutton) {
        console.error("Element #plot-time-domain not found!");
        return;
    }

    timebutton.addEventListener("click", function () { 
        timebutton.style.backgroundColor = "blue";
        timebutton.style.color = "white";
        
        if (!intervalid) { // Start the interval only if it's not already running
            isPaused = false;
            intervalid = setInterval(async () => {
                if (!isPaused) {
                    let data = null; // Initialize the data variable
                    try {
                        const response = await fetch(`${root}8000/api/latest_vibration`);
                        if (!response.ok) throw new Error("Network response was not ok");

                        const result = await response.json();
                        if (result.status === "success") {
                            data = result.data;
                        } else {
                            console.error("Error fetching data:", result.message);
                        }
                    } catch (error) {
                        console.error("Error fetching data:", error);
                    }

                    if (data) {
                        const labels = Array.from({ length: data.ax_data.length }, (_, i) => i + 1);

                        // Prepare datasets for the three axes (ax, ay, az)
                        const datasets = [
                            {
                                label: 'Accelerometer X-Axis',
                                data: data.ax_data,
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1,
                                pointRadius: 0.5,
                                fill: false
                            },
                            {
                                label: 'Accelerometer Y-Axis',
                                data: data.ay_data,
                                borderColor: 'rgba(153, 102, 255, 1)',
                                borderWidth: 1,
                                pointRadius: 0.5,
                                fill: false
                            },
                            {
                                label: 'Accelerometer Z-Axis',
                                data: data.az_data,
                                borderColor: 'rgba(255, 159, 64, 1)',
                                borderWidth: 1,
                                pointRadius: 0.5,
                                fill: false
                            }
                        ];

                        // Check if the data has changed
                        const currentData = JSON.stringify({ labels, datasets });
                        if (currentData === previousTimeData) {
                            console.log("No data change, skipping update.");
                            return; // Skip updating the chart if data hasn't changed
                        } else {
                            previousTimeData = currentData; // Update the stored data
                        }

                        // Plot the time-domain data
                        plotData(labels, datasets, 'Time-Domain Data');
                    }
                }
            }, 500); // Fetch and plot data every second
        }
    });
});

let chartInstancef = null; // Store the chart instance
let intervalId = null; // Store the interval ID
let isPaused = false; // Track the paused state
let previousData = null; // Store the previous data for comparison

// Function to fetch and plot data
async function fetchAndPlotData() {
    try {
        // Fetch the latest FFT data
        const response = await fetch(`${root}8000/api/latest_data`);
        if (!response.ok) {
            throw new Error("Failed to fetch FFT data.");
        }
        const fftData = await response.json();

        // Extract FFT data
        const { ax_freq, ax_fft, ay_freq, ay_fft, az_freq, az_fft } = fftData.data;

        // Check if the data has changed
        const currentData = JSON.stringify({ ax_freq, ax_fft, ay_freq, ay_fft, az_freq, az_fft });
        if (currentData === previousData) {
            console.log("No data change, skipping update.");
            return; // Skip updating the chart if data hasn't changed
        }
        previousData = currentData; // Update the stored data

        // Plot the FFT data for Ax, Ay, and Az
        plotFrequencyDomain(
            ax_freq, ax_fft,
            ay_freq, ay_fft,
            az_freq, az_fft,
            'Frequency-Domain Data (Ax, Ay, Az)'
        );
    } catch (error) {
        console.error("Error fetching or plotting frequency-domain data:", error);
    }
}




async function fetchAndplot3phaseV() {
    try {
        // Fetch the latest FFT data
        const response = await fetch(`${root}8000/api/latest_data`);
        if (!response.ok) {
            throw new Error("Failed to fetch FFT data.");
        }
        const fftData = await response.json();

        // Extract FFT data
        const { ax_freq, ax_fft, ay_freq, ay_fft, az_freq, az_fft } = fftData.data;

        // Check if the data has changed
        const currentData = JSON.stringify({ ax_freq, ax_fft, ay_freq, ay_fft, az_freq, az_fft });
        if (currentData === previousData) {
            console.log("No data change, skipping update.");
            return; // Skip updating the chart if data hasn't changed
        }
        previousData = currentData; // Update the stored data

        // Plot the FFT data for Ax, Ay, and Az
        plotFrequencyDomain(
            ax_freq, ax_fft,
            ay_freq, ay_fft,
            az_freq, az_fft,
            'Frequency-Domain Data (U, V, W)'
        );
    } catch (error) {
        console.error("Error fetching or plotting frequency-domain data:", error);
    }
}




// Function to plot frequency-domain data for Ax, Ay, and Az
function plotFrequencyDomain(axFreq, axData, ayFreq, ayData, azFreq, azData, title) {
    const ctx = document.getElementById("frequencyDomainChart").getContext("2d");

    // Destroy the existing chart if it exists
    if (chartInstancef) {
        chartInstancef.destroy();
    }

    // Create the new chart
    chartInstancef = new Chart(ctx, {
        type: 'line',
        data: {
            labels: axFreq, // Assuming all axes share the same frequency range
            datasets: [
                {
                    label: 'Ax FFT',
                    data: axData,
                    borderColor: 'red',
                    fill: false,
                    pointRadius: 0.5,
                    tension: 0,
                },
                {
                    label: 'Ay FFT',
                    data: ayData,
                    borderColor: 'green',
                    fill: false,
                    pointRadius: 0.5,
                    tension: 0,
                },
                {
                    label: 'Az FFT',
                    data: azData,
                    borderColor: 'blue',
                    fill: false,
                    pointRadius: 0.5,
                    tension: 0,
                }
            ]
        },
        options: {
            responsive: true,
            animation:false, // Disable animation for faster updates
            plugins: {
                title: {
                    display: true,
                    text: title
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Frequency (Hz)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'FFT Magnitude'
                    }
                }
            }
        }
    });
}

// Start fetching and plotting data when the button is clicked
document.addEventListener("DOMContentLoaded", function () {
    const freqbutton = document.getElementById("plot-frequency-domain");
    if (!freqbutton) {
        console.error("Element #plot-frequency-domain not found!");
        return;
    }

    freqbutton.addEventListener("click", function () {
        freqbutton.style.backgroundColor = "blue";
        freqbutton.style.color = "white";

        if (!intervalId) { // Start the interval only if it's not already running
            isPaused = false;
            intervalId = setInterval(() => {
                if (!isPaused) {
                    fetchAndPlotData();
                }
            }, 500); // Fetch and plot data every second
        }
    });
});


document.addEventListener("DOMContentLoaded", function () {
    const pauseButton = document.getElementById("pause-button");
    const resumeButton = document.getElementById("resume-button");

    if (!pauseButton || !resumeButton) {
        console.error("Pause or Resume button not found!");
        return;
    }

    // Pause fetching and plotting when the pause button is clicked
    pauseButton.addEventListener("click", function () {
        isPaused = true; // Set the paused state to true
        
        pauseButton.style.backgroundColor = "red";
        pauseButton.style.color = "white";

        resumeButton.style.backgroundColor = "gray";
        resumeButton.style.color = "white"; 
    });

    // Resume fetching and plotting when the resume button is clicked
    resumeButton.addEventListener("click", function () {
        isPaused = false; // Set the paused state to false

        resumeButton.style.backgroundColor = "green";
        resumeButton.style.color = "white";

        pauseButton.style.backgroundColor = "gray";
        pauseButton.style.color = "white"; 
    });
});
document.addEventListener("DOMContentLoaded", function () {
    const pauseButton = document.getElementById("pause-button");
    const resumeButton = document.getElementById("resume-button");

    if (!pauseButton || !resumeButton) {
        console.error("Pause or Resume button not found!");
        return;
    }

    // Pause fetching and plotting when the pause button is clicked
    pauseButton.addEventListener("click", function () {
        isPaused = true; // Set the paused state to true
        
        pauseButton.style.backgroundColor = "red";
        pauseButton.style.color = "white";

        resumeButton.style.backgroundColor = "gray";
        resumeButton.style.color = "white"; 
    });

    // Resume fetching and plotting when the resume button is clicked
    resumeButton.addEventListener("click", function () {
        isPaused = false; // Set the paused state to false

        resumeButton.style.backgroundColor = "green";
        resumeButton.style.color = "white";

        pauseButton.style.backgroundColor = "gray";
        pauseButton.style.color = "white"; 
    });
});








