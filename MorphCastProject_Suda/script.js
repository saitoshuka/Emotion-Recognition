console.log("Script loaded");

let myChart;

document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('emotionChart');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const statsConfig = {
        sendDatainterval: 5000,
        stopAfter: 7200000,
        licenseKey: "sk56f544483055ffe91e4e157cc750d6fb51943f6b7241"
    };

    const emotionLabelPlugin = {
        id: 'emotionLabelPlugin',
        afterDatasetDraw: (chart) => {
            const ctx = chart.ctx;
            chart.data.datasets.forEach((dataset, i) => {
                const meta = chart.getDatasetMeta(i);
                if (!meta.hidden) {
                    meta.data.forEach((element, index) => {
                        ctx.fillStyle = 'rgb(0, 0, 0)';
                        const fontSize = 12;
                        const fontStyle = 'normal';
                        const fontFamily = 'Helvetica Neue';
                        ctx.font = Chart.helpers.fontString(fontSize, fontStyle, fontFamily);
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'bottom';

                        const dataString = dataset.data[index].label;

                        ctx.fillText(dataString, element.x, element.y - 15);
                    });
                }
            });
        },
    };

    Chart.register(emotionLabelPlugin);

    const customTitlePlugin = {
        id: 'customTitlePlugin',
        afterDraw: (chart) => {
            const ctx = chart.ctx;
            const chartArea = chart.chartArea;
            ctx.save();

            // Draw custom y-axis title
            ctx.font = '11px Helvetica';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Valence', chartArea.left - 20, (chartArea.top + chartArea.bottom) / 2);

            // Draw custom x-axis title
            ctx.font = '11px Helvetica';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText('Arousal', (chartArea.left + chartArea.right) / 2, chartArea.bottom + 16);

            ctx.restore();
        },
    };

    Chart.register(customTitlePlugin);

    myChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Morphcast Facial Recognition',
                    data: [],
                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                    pointStyle: 'circle',
                },
                {
                    label: 'EEG Emotion Recognition',
                    data: [],
                    backgroundColor: 'rgba(255, 0, 0, 0.5)',
                    pointStyle: 'triangle',
                }
            ]
        },
        options: {
            scales: {
                x: {
                    type: 'linear',
                    position: 'center',
                    min: -1,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                    },
                    title: {
                        display: false,
                        text: 'Valence'
                    },
                    grid: {
                        drawBorder: true,
                        borderColor: 'black',
                        borderWidth: 1
                    }
                },
                y: {
                    type: 'linear',
                    position: 'center',
                    min: -1,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                    },
                    title: {
                        display: false,
                        text: 'Arousal'
                    },
                    grid: {
                        drawBorder: true,
                        borderColor: 'black',
                        borderWidth: 1
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    enabled: true
                }
            }
        }
    });

    const statisticsUploader = new MorphCastStatistics.StatisticsUploader(statsConfig);

    CY.loader()
        .licenseKey(statsConfig.licenseKey)
        .addModule(CY.modules().FACE_AROUSAL_VALENCE.name, { smoothness: 0.70 })
        .addModule(CY.modules().FACE_EMOTION.name, { smoothness: 0.40 })
        .addModule(CY.modules().FACE_ATTENTION.name, { smoothness: 0.83 })
        .addModule(CY.modules().FACE_AGE.name, { rawOutput: false })
        .addModule(CY.modules().FACE_GENDER.name, { smoothness: 0.95, threshold: 0.70 })
        .addModule(CY.modules().FACE_FEATURES.name, { smoothness: 0.90 })
        .addModule(CY.modules().FACE_POSITIVITY.name, { smoothness: 0.40, gain: 2, angle: 17 })
        .load()
        .then(async ({ start, stop }) => {
            await start();
            await statisticsUploader.start();

            setTimeout(async () => {
                await statisticsUploader.stop();
                await stop();
            }, statsConfig.stopAfter);
        })
        .catch((err) => {
            console.error('MorphCast init failed:', err);
            const box = document.getElementById('emotion-output');
            if (box) {
                box.textContent = `MorphCast init failed: ${err && err.message ? err.message : err}`;
            }
        });

    setupEventListeners();
    setupWebSocket();
});

function setupEventListeners() {
    window.addEventListener(CY.modules().FACE_AROUSAL_VALENCE.eventName, handleArousalValence);
    window.addEventListener(CY.modules().FACE_EMOTION.eventName, handleEmotion);
    window.addEventListener(CY.modules().FACE_GENDER.eventName, handleGender);
    window.addEventListener(CY.modules().FACE_AGE.eventName, handleAge);
    window.addEventListener(CY.modules().FACE_ATTENTION.eventName, handleFaceAttention);
    window.addEventListener(CY.modules().FACE_FEATURES.eventName, handleFaceFeatures);
    window.addEventListener(CY.modules().FACE_POSITIVITY.eventName, handlePositivity);
}

function setupWebSocket() {
    const socket = new WebSocket('ws://localhost:8767');

    socket.onopen = () => {
        console.log('WebSocket connection established.');
    };

    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Data received from WebSocket:', data);
        if (data.source === 'EEG') {
            console.log('Updating EEG data on the chart.');
            updateChart(myChart, data.valence, data.arousal, data.emotion, 'triangle', 'rgba(255, 0, 0, 0.5)', 1);
        } else {
            console.log('Updating Morphcast data on the chart.');
            updateChart(myChart, data.valence, data.arousal, data.emotion, 'circle', 'rgba(0, 123, 255, 0.5)', 0);
        }
        document.getElementById('valence-output').textContent = `Valence: ${data.valence}`;
        document.getElementById('arousal-output').textContent = `Arousal: ${data.arousal}`;
        document.getElementById('dominant-affect-output').textContent = `Emotion: ${data.emotion}`;
    };

    socket.onclose = () => {
        console.log('WebSocket connection closed.');
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function sendDataToServer(data) {
    console.log('Sending data to server:', data);  // Log data being sent
    fetch('http://localhost:5001/save_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
}

function handleArousalValence(evt) {
    const { valence, arousal, affects38 } = evt.detail.output || {};

    console.log(`Event Output: ${JSON.stringify(evt.detail.output)}`);
    if (!valence || !arousal || !affects38) return;

    console.log(`Affects38: ${JSON.stringify(affects38)}`);

    const [dominantAffect, affectValue] = Object.entries(affects38).reduce(
        (acc, [k, v]) => v > acc[1] ? [k, v] : acc,
        ["", 0]
    );

    const quadrant = mapValenceArousalToQuadrant(valence, arousal);

    updateChart(myChart, valence, arousal, dominantAffect, 'circle', 'rgba(0, 123, 255, 0.5)', 0);

    document.getElementById('valence-output').textContent = `Valence: ${valence}`;
    document.getElementById('arousal-output').textContent = `Arousal: ${arousal}`;
    document.getElementById('quadrant-output').textContent = `Quadrant: ${quadrant}`;
    document.getElementById('dominant-affect-output').textContent = `Dominant Affect: ${dominantAffect} (${affectValue})`;

    // Send data to the server
    const dataToSend = {
        valence: valence,
        arousal: arousal,
        dominant_affect: dominantAffect,
        quadrant: quadrant,
        affect_value: affectValue,
        attention_level: document.getElementById('attention-output').textContent.split(': ')[1],
        positivity_level: document.getElementById('positivity-output').textContent.split(': ')[1],
        gender: document.getElementById('gender-output').textContent.split(': ')[1],
        age: document.getElementById('age-output').textContent.split(': ')[1],
        facial_features: document.getElementById('features-output').textContent.split(': ')[1]
    };
    sendDataToServer(dataToSend);
}

function updateChart(chart, valence, arousal, emotionLabel, shape, color, datasetIndex) {
    console.log(`Updating chart - Dataset Index: ${datasetIndex}, Valence: ${valence}, Arousal: ${arousal}, Emotion: ${emotionLabel}`);
    const newData = { x: valence, y: arousal, label: emotionLabel };

    let backgroundColor;
    const quadrant = mapValenceArousalToQuadrant(valence, arousal);

    switch (quadrant) {
        case 'High Control':
            backgroundColor = 'yellow';
            break;
        case 'Obstructive':
            backgroundColor = 'red';
            break;
        case 'Low Control':
            backgroundColor = 'blue';
            break;
        case 'Conductive':
            backgroundColor = 'green';
            break;
        case 'Neutral':
            backgroundColor = 'gray';
            break;
        default:
            backgroundColor = color;
            break;
    }
    const size = 1 + Math.sqrt(Math.abs(valence) + Math.abs(arousal)) * 10;

    chart.data.datasets[datasetIndex].data = [newData];
    chart.data.datasets[datasetIndex].pointBackgroundColor = backgroundColor;
    chart.data.datasets[datasetIndex].pointRadius = size;
    chart.data.datasets[datasetIndex].pointStyle = shape; // Ensure the point style is set

    chart.update();
}

function mapValenceArousalToQuadrant(valence, arousal) {
    if (valence >= 0 && arousal >= 0) return 'High Control';
    if (valence < 0 && arousal >= 0) return 'Obstructive';
    if (valence < 0 && arousal < 0) return 'Low Control';
    if (valence >= 0 && arousal < 0) return 'Conductive';
    return 'Neutral';
}

function handleEmotion(evt) {
    const emotionData = evt.detail;
    if (emotionData) {
        console.log('Emotion Analysis:', emotionData);
        document.getElementById('emotion-output').textContent = `Detected Emotions: ${JSON.stringify(emotionData)}`;
    }
}

function handleGender(evt) {
    const data = evt.detail;
    if (data) {
        console.log('Gender Detection:', data);
        document.getElementById('gender-output').textContent = `Detected Gender: ${JSON.stringify(data)}`;
    }
}

function handleAge(evt) {
    const output = evt.detail.output;
    if (output) {
        const ageData = output.age;
        const numericAge = output.numericAge;

        console.log(`Age Groups: ${JSON.stringify(ageData)}, Numeric Age: ${numericAge}`);
        document.getElementById('age-output').textContent = `Detected Age: ${numericAge}`;
    }
}

function handleFaceAttention(evt) {
    const output = evt.detail.output;
    if (output) {
        const attention = output.attention;
        console.log(`Attention Level: ${attention}`);
        document.getElementById('attention-output').textContent = `Attention Level: ${attention}`;
    }
}

function handleFaceFeatures(evt) {
    const output = evt.detail.output;
    if (output) {
        const features = output.features;
        console.log('Facial Features:', features);
        document.getElementById('features-output').textContent = `Facial Features: ${JSON.stringify(features)}`;
    }
}

function handlePositivity(evt) {
    const output = evt.detail.output;
    if (output) {
        const positivity = output.positivity;
        console.log('Positivity:', positivity);
        document.getElementById('positivity-output').textContent = `Positivity Level: ${positivity}`;
    }
}

// ===== Override for compare-server compatibility =====
let wsSocket = null;

function setupWebSocket() {
    wsSocket = new WebSocket('ws://localhost:8767');

    wsSocket.onopen = () => {
        console.log('WebSocket connection established.');
    };

    wsSocket.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        const data = msg && msg.type === 'update' ? msg.data : msg;
        if (!data || !data.source) return;

        console.log('Data received from WebSocket:', data);
        if (String(data.source).toLowerCase() === 'eeg') {
            updateChart(myChart, data.valence, data.arousal, data.emotion || 'EEG', 'triangle', 'rgba(255, 0, 0, 0.5)', 1);
            document.getElementById('dominant-affect-output').textContent = `EEG Emotion: ${data.emotion || ''}`;
        } else if (String(data.source).toLowerCase() === 'morphcast') {
            updateChart(myChart, data.valence, data.arousal, data.emotion || 'Morphcast', 'circle', 'rgba(0, 123, 255, 0.5)', 0);
        }
    };

    wsSocket.onclose = () => {
        console.log('WebSocket connection closed.');
    };

    wsSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function handleArousalValence(evt) {
    const { valence, arousal, affects38 } = evt.detail.output || {};
    if (valence === undefined || arousal === undefined) return;

    const [dominantAffect, affectValue] = Object.entries(affects38 || {}).reduce(
        (acc, [k, v]) => (v > acc[1] ? [k, v] : acc),
        ["neutral", 0]
    );

    const quadrant = mapValenceArousalToQuadrant(valence, arousal);
    updateChart(myChart, valence, arousal, dominantAffect, 'circle', 'rgba(0, 123, 255, 0.5)', 0);

    document.getElementById('valence-output').textContent = `Valence: ${valence}`;
    document.getElementById('arousal-output').textContent = `Arousal: ${arousal}`;
    document.getElementById('quadrant-output').textContent = `Quadrant: ${quadrant}`;
    document.getElementById('dominant-affect-output').textContent = `Dominant Affect: ${dominantAffect} (${affectValue})`;

    // Send Morphcast point to websocket compare server
    if (wsSocket && wsSocket.readyState === WebSocket.OPEN) {
        wsSocket.send(JSON.stringify({
            source: 'Morphcast',
            valence: Number(valence),
            arousal: Number(arousal),
            emotion: String(dominantAffect || 'Morphcast')
        }));
    }
}


