import os
import joblib
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Safe model loader
MODEL_PATH = "adaboost_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customizable AI Intelligence Portal</title>
    
    <!-- Dynamic Google Fonts -->
    <link id="font-stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style id="custom-styles">
        :root {
            --bg-dark: #05050d;
            --card-bg: rgba(13, 12, 29, 0.82);
            --card-border: rgba(186, 85, 211, 0.25);
            --accent-primary: #ff007f;
            --accent-secondary: #a855f7;
            --accent-cyan: #00f0ff;
            --accent-emerald: #10b981;
            --text-main: #f3f4f6;
            --text-muted: #a1a1aa;
            --input-bg: rgba(255, 255, 255, 0.04);
            --input-border: rgba(168, 85, 247, 0.3);
            --font-family: 'Space Grotesk', sans-serif;
            --blur-intensity: 25px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: var(--font-family);
            transition: background 0.3s ease, border-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }

        body {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow-x: hidden;
            position: relative;
            padding: 40px 20px;
            background: var(--bg-dark);
            color: var(--text-main);
        }

        /* Canvas Overlay for Particle / Matrix Effects */
        #bg-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }

        .container {
            position: relative;
            z-index: 10;
            width: 100%;
            max-width: 1100px;
            background: var(--card-bg);
            backdrop-filter: blur(var(--blur-intensity));
            -webkit-backdrop-filter: blur(var(--blur-intensity));
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 0 50px rgba(0, 0, 0, 0.5);
        }

        /* Style Customizer Toolbar Panel */
        .style-toolbar {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 16px 20px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            align-items: center;
        }

        .toolbar-title {
            grid-column: 1 / -1;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-cyan);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header .badge {
            display: inline-block;
            padding: 6px 18px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            border-radius: 30px;
            color: var(--accent-cyan);
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--text-main) 0%, var(--accent-cyan) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 18px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .form-group label i {
            color: var(--accent-cyan);
        }

        .form-control {
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 12px;
            padding: 12px 16px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
        }

        .form-control:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px var(--accent-cyan);
        }

        select.form-control option {
            background-color: #0d0c1d;
            color: #ffffff;
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 15px;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 18px;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 1px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            box-shadow: 0 10px 25px -5px var(--accent-primary);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            filter: brightness(1.15);
        }

        /* Analytics Output Card */
        .result-card {
            margin-top: 35px;
            padding: 30px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            display: none;
            animation: slideUp 0.5s ease forwards;
        }

        .result-header {
            text-align: center;
            margin-bottom: 25px;
        }

        .result-title {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }

        .result-value {
            font-size: 1.8rem;
            font-weight: 700;
            display: inline-block;
            padding: 12px 28px;
            border-radius: 14px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--card-border);
        }

        .result-positive {
            color: var(--accent-emerald);
            border-color: var(--accent-emerald);
            box-shadow: 0 0 25px var(--accent-emerald);
        }

        .result-negative {
            color: var(--accent-primary);
            border-color: var(--accent-primary);
            box-shadow: 0 0 25px var(--accent-primary);
        }

        .analytics-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 25px;
            margin-top: 25px;
            align-items: center;
        }

        @media (min-width: 850px) {
            .analytics-grid {
                grid-template-columns: 1.1fr 0.9fr;
            }
        }

        .chart-box {
            position: relative;
            width: 100%;
            height: 300px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            border: 1px solid var(--card-border);
        }

        .metrics-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .metric-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 16px;
        }

        .metric-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            display: flex;
            justify-content: space-between;
        }

        .metric-bar-bg {
            height: 10px;
            width: 100%;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
        }

        .metric-bar-fill {
            height: 100%;
            width: 0%;
            border-radius: 5px;
            transition: width 1s ease;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <canvas id="bg-canvas"></canvas>

    <div class="container">
        
        <!-- Live Style & Color Customizer Dropdown Bar -->
        <div class="style-toolbar">
            <div class="toolbar-title">
                <i class="fa-solid fa-palette"></i> Dynamic Theme Customizer
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-droplet"></i> Color Palette</label>
                <select id="themeSelector" class="form-control" onchange="applyTheme(this.value)">
                    <option value="cyber">Cyber Synthwave (Neon Purple/Pink)</option>
                    <option value="darkGlass">Executive Glass (Deep Navy/Cyan)</option>
                    <option value="emerald">Emerald Matrix (Dark Green/Mint)</option>
                    <option value="sunset">Sunset Crimson (Obsidian/Orange)</option>
                    <option value="gold">Royal Gold (Dark Ebony/Gold)</option>
                </select>
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-wand-magic-sparkles"></i> Background Style</label>
                <select id="bgSelector" class="form-control" onchange="applyBackground(this.value)">
                    <option value="particles">Particle Constellation</option>
                    <option value="matrix">Digital Rain (Matrix)</option>
                    <option value="gradient">Glowing Gradient Wave</option>
                    <option value="flat">Static Glassmorphism</option>
                </select>
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-layer-group"></i> Glassmorphism Blur</label>
                <select id="blurSelector" class="form-control" onchange="applyBlur(this.value)">
                    <option value="25px">Heavy Blur (25px)</option>
                    <option value="12px">Medium Blur (12px)</option>
                    <option value="5px">Light Glass (5px)</option>
                    <option value="0px">Flat Solid (0px)</option>
                </select>
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-font"></i> Font Family</label>
                <select id="fontSelector" class="form-control" onchange="applyFont(this.value)">
                    <option value="Space Grotesk">Space Grotesk (Tech)</option>
                    <option value="Plus Jakarta Sans">Plus Jakarta Sans (Corporate)</option>
                    <option value="Inter">Inter (Clean)</option>
                    <option value="Roboto Mono">Roboto Mono (Monospace)</option>
                </select>
            </div>
        </div>

        <div class="header">
            <span class="badge"><i class="fa-solid fa-microchip"></i> Machine Learning Engine</span>
            <h1>AdaBoost Prediction Portal</h1>
            <p>Select your favorite UI style above and execute real-time model analytics</p>
        </div>

        <form id="prediction-form" class="grid-form">
            <div class="form-group">
                <label><i class="fa-solid fa-user"></i> Age</label>
                <input type="number" name="Age" class="form-control" placeholder="e.g. 34" required min="18" max="100" value="34">
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-venus-mars"></i> Gender</label>
                <select name="Gender" class="form-control" required>
                    <option value="0">Female</option>
                    <option value="1" selected>Male</option>
                </select>
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-calendar-check"></i> Tenure (Months)</label>
                <input type="number" name="Tenure" class="form-control" placeholder="e.g. 12" required min="0" value="12">
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-bolt"></i> Usage Frequency</label>
                <input type="number" name="Usage Frequency" class="form-control" placeholder="e.g. 18" required min="0" value="18">
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-headset"></i> Support Calls</label>
                <input type="number" name="Support Calls" class="form-control" placeholder="e.g. 2" required min="0" value="2">
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-clock"></i> Payment Delay (Days)</label>
                <input type="number" name="Payment Delay" class="form-control" placeholder="e.g. 5" required min="0" value="5">
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-tags"></i> Subscription Type</label>
                <select name="Subscription Type" class="form-control" required>
                    <option value="0">Basic</option>
                    <option value="1" selected>Standard</option>
                    <option value="2">Premium</option>
                </select>
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-file-contract"></i> Contract Length</label>
                <select name="Contract Length" class="form-control" required>
                    <option value="0">Monthly</option>
                    <option value="1" selected>Quarterly</option>
                    <option value="2">Annual</option>
                </select>
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-dollar-sign"></i> Total Spend ($)</label>
                <input type="number" step="0.01" name="Total Spend" class="form-control" placeholder="e.g. 850.50" required min="0" value="850">
            </div>

            <div class="form-group">
                <label><i class="fa-solid fa-hand-pointer"></i> Last Interaction (Days)</label>
                <input type="number" name="Last Interaction" class="form-control" placeholder="e.g. 14" required min="0" value="14">
            </div>

            <button type="submit" class="submit-btn">
                <i class="fa-solid fa-play"></i> Execute Prediction Analytics
            </button>
        </form>

        <div id="result-box" class="result-card">
            <div class="result-header">
                <div class="result-title">Model Decision Output</div>
                <div id="result-text" class="result-value">---</div>
            </div>

            <div class="analytics-grid">
                <div class="chart-box">
                    <canvas id="radarChart"></canvas>
                </div>

                <div class="metrics-panel">
                    <div class="metric-item">
                        <div class="metric-label">
                            <span>Risk Score Index</span>
                            <span id="riskPctText">0%</span>
                        </div>
                        <div class="metric-bar-bg">
                            <div id="riskBar" class="metric-bar-fill" style="background: var(--accent-primary);"></div>
                        </div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-label">
                            <span>Platform Usage Index</span>
                            <span id="usagePctText">0%</span>
                        </div>
                        <div class="metric-bar-bg">
                            <div id="engagementBar" class="metric-bar-fill" style="background: var(--accent-cyan);"></div>
                        </div>
                    </div>

                    <div class="metric-item">
                        <div class="metric-label">
                            <span>Customer Value Metric</span>
                            <span id="spendPctText">0%</span>
                        </div>
                        <div class="metric-bar-bg">
                            <div id="spendBar" class="metric-bar-fill" style="background: var(--accent-secondary);"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Color Theme Presets Config
        const themePresets = {
            cyber: {
                bgDark: '#05050d',
                cardBg: 'rgba(13, 12, 29, 0.82)',
                cardBorder: 'rgba(186, 85, 211, 0.25)',
                primary: '#ff007f',
                secondary: '#a855f7',
                cyan: '#00f0ff',
                emerald: '#10b981'
            },
            darkGlass: {
                bgDark: '#0a0e17',
                cardBg: 'rgba(16, 24, 40, 0.80)',
                cardBorder: 'rgba(0, 242, 254, 0.25)',
                primary: '#f43f5e',
                secondary: '#7f00ff',
                cyan: '#00f2fe',
                emerald: '#10b981'
            },
            emerald: {
                bgDark: '#021810',
                cardBg: 'rgba(6, 38, 25, 0.82)',
                cardBorder: 'rgba(16, 185, 129, 0.3)',
                primary: '#f43f5e',
                secondary: '#059669',
                cyan: '#34d399',
                emerald: '#10b981'
            },
            sunset: {
                bgDark: '#1a0c0c',
                cardBg: 'rgba(38, 16, 16, 0.82)',
                cardBorder: 'rgba(249, 115, 22, 0.3)',
                primary: '#ef4444',
                secondary: '#f97316',
                cyan: '#facc15',
                emerald: '#10b981'
            },
            gold: {
                bgDark: '#12100b',
                cardBg: 'rgba(31, 26, 16, 0.82)',
                cardBorder: 'rgba(234, 179, 8, 0.3)',
                primary: '#e11d48',
                secondary: '#eab308',
                cyan: '#fef08a',
                emerald: '#10b981'
            }
        };

        function applyTheme(themeKey) {
            const t = themePresets[themeKey];
            const root = document.documentElement;
            root.style.setProperty('--bg-dark', t.bgDark);
            root.style.setProperty('--card-bg', t.cardBg);
            root.style.setProperty('--card-border', t.cardBorder);
            root.style.setProperty('--accent-primary', t.primary);
            root.style.setProperty('--accent-secondary', t.secondary);
            root.style.setProperty('--accent-cyan', t.cyan);
            root.style.setProperty('--accent-emerald', t.emerald);
        }

        function applyBlur(blurVal) {
            document.documentElement.style.setProperty('--blur-intensity', blurVal);
        }

        function applyFont(fontName) {
            document.documentElement.style.setProperty('--font-family', `'${fontName}', sans-serif`);
        }

        // Canvas Background Renderers (Particles, Matrix, Gradient)
        const canvas = document.getElementById('bg-canvas');
        const ctx = canvas.getContext('2d');
        let activeBgMode = 'particles';
        let animationFrameId = null;

        function resizeCanvas() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        // Particles Mode
        const particles = Array.from({ length: 45 }, () => ({
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            size: Math.random() * 2 + 1,
            speedX: (Math.random() - 0.5) * 0.8,
            speedY: (Math.random() - 0.5) * 0.8
        }));

        function drawParticles() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            particles.forEach(p => {
                p.x += p.speedX;
                p.y += p.speedY;
                if (p.x < 0 || p.x > canvas.width) p.speedX *= -1;
                if (p.y < 0 || p.y > canvas.height) p.speedY *= -1;

                ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent-cyan').trim() || '#00f0ff';
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
            });
            animationFrameId = requestAnimationFrame(drawParticles);
        }

        // Matrix Mode
        const matrixChars = '011010101001MLAI';
        const fontSize = 14;
        let columns = Math.floor(window.innerWidth / fontSize);
        let drops = Array(columns).fill(1);

        function drawMatrix() {
            ctx.fillStyle = 'rgba(5, 5, 13, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent-cyan').trim() || '#00f0ff';
            ctx.font = fontSize + 'px monospace';

            drops.forEach((y, i) => {
                const text = matrixChars.charAt(Math.floor(Math.random() * matrixChars.length));
                ctx.fillText(text, i * fontSize, y * fontSize);
                if (y * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
                drops[i]++;
            });
            animationFrameId = requestAnimationFrame(drawMatrix);
        }

        function applyBackground(mode) {
            cancelAnimationFrame(animationFrameId);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            document.body.style.animation = 'none';

            if (mode === 'particles') {
                drawParticles();
            } else if (mode === 'matrix') {
                columns = Math.floor(window.innerWidth / fontSize);
                drops = Array(columns).fill(1);
                drawMatrix();
            } else if (mode === 'gradient') {
                document.body.style.background = 'linear-gradient(-45deg, #05050d, #1a0b2e, #2b0938, #031329)';
                document.body.style.backgroundSize = '400% 400%';
                document.body.style.animation = 'gradientWave 10s ease infinite';
            } else {
                document.body.style.background = 'var(--bg-dark)';
            }
        }

        // Initial launch
        drawParticles();

        // Chart.js Radar Instance
        let radarChartInstance = null;

        function renderRadarChart(data) {
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-cyan').trim() || '#00f0ff';

            const normalizedData = [
                Math.min(100, (data['Age'] / 80) * 100),
                Math.min(100, (data['Tenure'] / 60) * 100),
                Math.min(100, (data['Usage Frequency'] / 30) * 100),
                Math.min(100, (data['Support Calls'] / 10) * 100),
                Math.min(100, (data['Payment Delay'] / 30) * 100),
                Math.min(100, (data['Total Spend'] / 2000) * 100)
            ];

            if (radarChartInstance) radarChartInstance.destroy();

            radarChartInstance = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Age', 'Tenure', 'Usage', 'Calls', 'Delay', 'Spend'],
                    datasets: [{
                        label: 'Feature Intensity',
                        data: normalizedData,
                        backgroundColor: 'rgba(0, 240, 255, 0.2)',
                        borderColor: primaryColor,
                        borderWidth: 2,
                        pointBackgroundColor: primaryColor
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: 'rgba(255, 255, 255, 0.15)' },
                            grid: { color: 'rgba(255, 255, 255, 0.15)' },
                            ticks: { display: false },
                            suggestedMin: 0,
                            suggestedMax: 100
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        // AJAX Form Processing
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {};
            formData.forEach((value, key) => data[key] = parseFloat(value));

            const resultBox = document.getElementById('result-box');
            const resultText = document.getElementById('result-text');

            resultBox.style.display = 'block';
            resultText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const res = await response.json();
                if (res.status === 'success') {
                    resultText.className = 'result-value ' + (res.prediction === 1 ? 'result-negative' : 'result-positive');
                    resultText.innerHTML = res.label;

                    renderRadarChart(data);

                    const riskPct = res.prediction === 1 ? 88 : 12;
                    const usagePct = Math.round(Math.min(100, (data['Usage Frequency'] / 30) * 100));
                    const spendPct = Math.round(Math.min(100, (data['Total Spend'] / 2000) * 100));

                    setTimeout(() => {
                        document.getElementById('riskBar').style.width = riskPct + '%';
                        document.getElementById('engagementBar').style.width = usagePct + '%';
                        document.getElementById('spendBar').style.width = spendPct + '%';

                        document.getElementById('riskPctText').innerText = riskPct + '%';
                        document.getElementById('usagePctText').innerText = usagePct + '%';
                        document.getElementById('spendPctText').innerText = spendPct + '%';
                    }, 100);
                } else {
                    resultText.innerText = 'Error: ' + res.message;
                }
            } catch (err) {
                resultText.innerText = 'Execution Failed. Check server logs.';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'status': 'error', 
            'message': 'Model file (adaboost_model.pkl) was not found or failed to load on the server.'
        }), 500

    try:
        data = request.get_json(force=True)
        
        feature_order = [
            'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
            'Payment Delay', 'Subscription Type', 'Contract Length', 
            'Total Spend', 'Last Interaction'
        ]
        
        features = []
        for feat in feature_order:
            val = data.get(feat)
            if val is None:
                return jsonify({
                    'status': 'error', 
                    'message': f'Missing value for feature: "{feat}"'
                }), 400
            features.append(float(val))

        input_array = np.array([features])
        prediction = int(model.predict(input_array)[0])
        
        label = "High Risk / Churn Predicted" if prediction == 1 else "Low Risk / Active Customer"

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'label': label
        })

    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': f'Prediction Error: {str(e)}'
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
