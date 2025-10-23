const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const scanBtn = document.getElementById('scanBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const resultAge = document.getElementById('resultAge');
const ageGroup = document.getElementById('ageGroup');
const ageGroupConfidence = document.getElementById('ageGroupConfidence');
const gender = document.getElementById('gender');
const genderConfidence = document.getElementById('genderConfidence');

let selectedFile = null;

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        fileName.textContent = file.name;
        scanBtn.disabled = false;

        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewSection.style.display = 'block';
            resultSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
});

scanBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    resultSection.style.display = 'none';
    loading.style.display = 'block';
    scanBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch('/predict-age', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded ${response.status}`);
        }

        const data = await response.json();
        loading.style.display = 'none';

        if (data.success) {
            resultAge.textContent = `${data.age} year olds`;
            ageGroup.textContent = data.age_group;
            ageGroupConfidence.textContent = `Confidence: ${data.age_group_confidence}%`;
            gender.textContent = data.gender;
            gender.className = data.gender === 'Male'
                ? 'detail-value gender-male'
                : 'detail-value gender-female';
            genderConfidence.textContent = `Confidence: ${data.gender_confidence}%`;
            resultSection.style.display = 'block';
        } else {
            alert('Error: ' + (data.message || 'Prediction failed'));
        }
    } catch (error) {
        loading.style.display = 'none';
        alert('Cannot connect to server: ' + error.message);
    } finally {
        scanBtn.disabled = false;
    }
});
