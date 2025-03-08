let fftChart, adc1Chart, adc2Chart; // Global chart variables
let updateInterval; // Stores the interval for fetching data
let isPaused = false; // Track the paused state
const root = config.ROOT_URL // Read from config.js
const ports = [8000, 8001, 8002, 8003, 9000, 9001, 9002]; // List of potential ports

document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    setupButtons();
    startUpdating();
});

// Function to initialize all charts
function initializeCharts() {
    const fftCtx = document.getElementById('fftChart').getContext('2d');
    fftChart = new Chart(fftCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            animation: false,
            scales: {
                x: { title: { display: true, text: 'Frequency (Hz)' } },
                y: { title: { display: true, text: 'Magnitude' } }
            }
        }
    });

    const adc1Ctx = document.getElementById('adc1Chart').getContext('2d');
    adc1Chart = new Chart(adc1Ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { title: { display: true, text: 'Sample Index' } },
                y: { title: { display: true, text: 'VOLTAGE' } }
            }
        }
    });

    const adc2Ctx = document.getElementById('adc2Chart').getContext('2d');
    adc2Chart = new Chart(adc2Ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { title: { display: true, text: 'Sample Index' } },
                y: { title: { display: true, text: 'CURRENT' } }
            }
        }
    });
}

// Function to calculate power factor
/*function calculatePowerFactor(voltage, current) {
      voltage=voltage-2070;
      current=current -2070;
    const crossProduct = voltage.reduce((sum, val, index) => sum + val * current[index], 0);
    const voltageSum = voltage.reduce((sum, val) => sum + val * val, 0);
    const currentSum = current.reduce((sum, val) => sum + val * val, 0);
    return crossProduct / (Math.sqrt(voltageSum) * Math.sqrt(currentSum));
}*/

function calculatePowerFactor(voltage, current) {
    // Subtract 2070 from each element in the voltage and current arrays
   
    // Calculate cross product, voltage sum, and current sum
    const crossProduct = voltage.reduce((sum, val, index) => sum + val * current[index], 0);
    const voltageSum = voltage.reduce((sum, val) => sum + val * val, 0);
    const currentSum = current.reduce((sum, val) => sum + val * val, 0);

    // Calculate and return the power factor
    return crossProduct / (Math.sqrt(voltageSum) * Math.sqrt(currentSum));
}

// Function to fetch ADC Data
async function fetchADCData() {
    if (isPaused) return; // Stop updating if paused
    const result = await findAvailablePort('/api/adc1');
    if (result && result.status === "success" && result.data && Array.isArray(result.data.data)) {
        const adcData = result.data.data;
        const sampleIndices = adcData.map((_, index) => index);
        const channels = transpose(adcData);

        // Update ADC1 Chart (Channels 1-3)
        adc1Chart.data.datasets = [];
        adc1Chart.data.labels = sampleIndices;
        for (let i = 0; i < 3; i++) {
            adc1Chart.data.datasets.push({
                label: `Channel ${i + 1}`,
                data: channels[i],
                borderColor: colors[i],
                backgroundColor: 'rgba(0, 123, 255, 0.2)',
                fill: false,
                tension: 0.4,
            });
        }
        adc1Chart.update();

        // Update ADC2 Chart (Channels 4-6)
        adc2Chart.data.datasets = [];
        adc2Chart.data.labels = sampleIndices;
        for (let i = 3; i < 6; i++) {
            adc2Chart.data.datasets.push({
                label: `Channel ${i + 1}`,
                data: channels[i],
                borderColor: colors[i],
                backgroundColor: 'rgba(0, 123, 255, 0.2)',
                fill: false,
                tension: 0.4,
            });
        }
        adc2Chart.update();

        // Calculate and display power factors
        const powerFactorPhase1 = calculatePowerFactor(channels[2], channels[3]); // Channels 3 & 4
        const powerFactorPhase2 = calculatePowerFactor(channels[0], channels[4]); // Channels 1 & 6
        const powerFactorPhase3 = calculatePowerFactor(channels[1], channels[5]); // Channels 2 & 5

        document.getElementById('power-factor-phase1').textContent = powerFactorPhase1.toFixed(2);
        document.getElementById('power-factor-phase2').textContent = powerFactorPhase2.toFixed(2);
        document.getElementById('power-factor-phase3').textContent = powerFactorPhase3.toFixed(2);
    }
}

// Function to fetch FFT Data
async function fetchFFTData() {
    if (isPaused) return; // Stop updating if paused
    const result = await findAvailablePort('/api/adc');
    if (result && result.data) {
        const channelData = result.data;
        fftChart.data.datasets = [];

        for (let i = 1; i <= 6; i++) {
            const channel = channelData[`channel_${i}`];
            if (channel) {
                const filteredData = channel.frequency
                    .map((freq, index) => ({ freq, mag: channel.magnitude[index] }))
                    .filter(({ freq }) => freq >= 40 && freq <= 1000);

                if (i === 1) fftChart.data.labels = filteredData.map(({ freq }) => freq);

                fftChart.data.datasets.push({
                    label: `Channel ${i}`,
                    data: filteredData.map(({ mag }) => mag),
                    borderColor: colors[i - 1],
                    backgroundColor: 'rgba(0, 123, 255, 0.2)',
                    fill: false,
                    tension: 0.4,
                });
            }
        }
        fftChart.update();
    }
}

// Function to find the first available server port
async function findAvailablePort(endpoint) {
    const urls = ports.map(port => `${root}${port}${endpoint}`);
    try {
        const response = await Promise.any(urls.map(url => fetch(url).then(res => res.ok ? res.json() : Promise.reject(res))));
        return response;
    } catch (error) {
        console.error(`All fetch attempts failed for ${endpoint}:`, error);
        return null;
    }
}

// Function to transpose data
function transpose(data) {
    return data[0].map((_, colIndex) => data.map(row => row[colIndex]));
}

// Predefined colors
const colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33F3FF'];

// Function to start data fetching
function startUpdating() {
    clearInterval(updateInterval); // Clear any existing intervals
    updateInterval = setInterval(() => {
        fetchADCData();
        fetchFFTData();
    }, 1000); // Update every 1 second
}

// Function to set up stop and resume buttons
function setupButtons() {
    const stopButton = document.getElementById("stopButton");
    const resumeButton = document.getElementById("resumeButton");

    stopButton.addEventListener("click", function () {
        isPaused = true;
        stopButton.style.backgroundColor = "red";
        stopButton.style.color = "white";
        resumeButton.style.backgroundColor = "gray";
        resumeButton.style.color = "white";
    });

    resumeButton.addEventListener("click", function () {
        isPaused = false;
        resumeButton.style.backgroundColor = "#28a745"; // Green
        resumeButton.style.color = "white";
        stopButton.style.backgroundColor = "gray";
        stopButton.style.color = "white";
        startUpdating(); // Restart updates when resumed
    });
}