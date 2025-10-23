const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = 3000;

// Multer setup
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const dir = './uploads';
        if (!fs.existsSync(dir)) fs.mkdirSync(dir);
        cb(null, dir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({
    storage,
    fileFilter: (req, file, cb) => {
        const allowed = /jpeg|jpg|png|gif/;
        const isValid = allowed.test(file.mimetype) && allowed.test(path.extname(file.originalname).toLowerCase());
        isValid ? cb(null, true) : cb(new Error('Only image files are allowed!'));
    }
});

// Express config
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static('public'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => res.render('index'));

app.post('/predict-age', upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ success: false, message: 'No image uploaded' });

    const imagePath = req.file.path;
    console.log(`Processing image: ${imagePath}`);

    try {
        const result = await runPythonModel(imagePath);
        fs.unlinkSync(imagePath);

        res.json({ success: true, ...result });
    } catch (err) {
        console.error(err);
        if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath);
        res.status(500).json({ success: false, message: err.message });
    }
});

function runPythonModel(imagePath) {
    return new Promise((resolve, reject) => {
        const process = spawn('python3', ['predict.py', imagePath]);
        let output = '', errorOutput = '';

        process.stdout.on('data', (data) => output += data.toString());
        process.stderr.on('data', (data) => errorOutput += data.toString());

        process.on('close', (code) => {
            if (code !== 0) return reject(new Error(errorOutput || 'Python script failed'));

            try {
                const parsed = JSON.parse(output);
                if (parsed.error) reject(new Error(parsed.error));
                else resolve(parsed);
            } catch {
                reject(new Error('Invalid Python output: ' + output));
            }
        });

        setTimeout(() => {
            process.kill();
            reject(new Error('Prediction timeout'));
        }, 30000);
    });
}

// Start server
app.listen(PORT, () => {
    console.log(`   Server running at http://localhost:${PORT}`);
});
