/**
 * Product Defect Detection System - Frontend JavaScript
 * Handles file upload, API calls, and dynamic UI updates
 */

// ============================================
// DOM Elements
// ============================================
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const actionButtons = document.getElementById('actionButtons');
const uploadAnotherBtn = document.getElementById('uploadAnotherBtn');
const downloadReportBtn = document.getElementById('downloadReportBtn');

// Image elements
const originalPlaceholder = document.getElementById('originalPlaceholder');
const heatmapPlaceholder = document.getElementById('heatmapPlaceholder');
const originalImage = document.getElementById('originalImage');
const heatmapImage = document.getElementById('heatmapImage');

// Results elements
const statusBadge = document.getElementById('statusBadge');
const severityBadge = document.getElementById('severityBadge');
const severityMarker = document.getElementById('severityMarker');
const severityValue = document.getElementById('severityValue');
//const inferenceTime = document.getElementById('inferenceTime');

// State
let selectedFile = null;
let lastResult = null;

// ============================================
// Event Listeners
// ============================================

// Click to upload
uploadArea.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop events
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Remove preview
removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelection();
});

// Analyze button
analyzeBtn.addEventListener('click', analyzeImage);

// Upload another button
uploadAnotherBtn.addEventListener('click', () => {
    clearSelection();
    resetResults();
});

// Download report button
downloadReportBtn.addEventListener('click', downloadReport);

// ============================================
// Functions
// ============================================

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        alert('Please select a valid image file (JPG, PNG, BMP, or GIF)');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        alert('File too large. Maximum size is 16MB');
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        uploadArea.style.display = 'none';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearSelection() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewContainer.style.display = 'none';
    uploadArea.style.display = 'block';
    analyzeBtn.disabled = true;
}

function resetResults() {
    lastResult = null;

    // Hide images, show placeholders
    originalImage.style.display = 'none';
    heatmapImage.style.display = 'none';
    originalPlaceholder.style.display = 'flex';
    heatmapPlaceholder.style.display = 'flex';

    // Reset status
    statusBadge.className = 'status-badge waiting';
    statusBadge.innerHTML = '<span class="status-icon">⏳</span><span class="status-text">WAITING FOR IMAGE</span>';

    // Hide defect type row
    const defectTypeRow = document.getElementById('defectTypeRow');
    if (defectTypeRow) {
        defectTypeRow.style.display = 'none';
    }

    // Reset metrics
    severityBadge.textContent = '--';
    severityBadge.className = 'severity-badge';
    severityMarker.style.left = '0%';
    severityValue.textContent = '-- / 100';
   // inferenceTime.textContent = '-- ms';

    // Disable download button, hide upload another, hide results section
    downloadReportBtn.disabled = true;
    uploadAnotherBtn.style.display = 'none';
    resultsSection.style.display = 'none';
}

async function analyzeImage() {
    if (!selectedFile) {
        alert('Please select an image first');
        return;
    }

    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoading = analyzeBtn.querySelector('.btn-loading');
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-flex';
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            lastResult = data;
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error occurred'));
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please try again.');
    } finally {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function displayResults(data) {
    // Hide placeholders, show images
    originalPlaceholder.style.display = 'none';
    heatmapPlaceholder.style.display = 'none';
    originalImage.style.display = 'block';
    heatmapImage.style.display = 'block';

    // Set images
    originalImage.src = data.original_url;
    heatmapImage.src = data.heatmap_url || data.original_url;

    // Get defect type elements
    const defectTypeRow = document.getElementById('defectTypeRow');
    const defectClass = document.getElementById('defectClass');

    // Set status badge and defect type
    if (data.is_defective) {
        statusBadge.className = 'status-badge defect';
        statusBadge.innerHTML = '<span class="status-icon">✕</span><span class="status-text">DEFECT DETECTED</span>';

        // Show and set defect type
        defectTypeRow.style.display = 'flex';
        defectClass.textContent = data.predicted_class;
    } else {
        statusBadge.className = 'status-badge no-defect';
        statusBadge.innerHTML = '<span class="status-icon">✓</span><span class="status-text">NO DEFECT</span>';

        // Hide defect type for non-defective items
        defectTypeRow.style.display = 'none';
    }

    // Set confidence
    // Confidence display removed by request

    // Set severity
    const severity = data.severity_score || 0;
    const severityLevel = data.severity_level || 'LOW';

    severityBadge.textContent = severityLevel;
    severityBadge.className = 'severity-badge ' + severityLevel.toLowerCase();

    severityValue.textContent = severity + ' / 100';
    animateSeverityGauge(severity);

    // Set inference time
    //inferenceTime.textContent = (data.inference_time_ms || 0) + ' ms';

    // Show results section, enable download button, show upload another
    resultsSection.style.display = 'block';
    downloadReportBtn.disabled = false;
    uploadAnotherBtn.style.display = 'block';
}



function animateSeverityGauge(value) {
    setTimeout(() => {
        severityMarker.style.left = Math.min(value, 98) + '%';
    }, 100);
}

async function downloadReport() {
    if (!lastResult) {
        alert('No analysis results to download');
        return;
    }

    const originalText = downloadReportBtn.innerHTML;
    downloadReportBtn.innerHTML = '<span class="spinner"></span> Generating...';
    downloadReportBtn.disabled = true;

    try {
        const response = await fetch('/api/download-report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(lastResult)
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'defect_report_' + new Date().toISOString().slice(0, 10) + '.pdf';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const data = await response.json();
            alert('Error: ' + (data.error || 'Failed to generate report'));
        }

    } catch (error) {
        console.error('Error:', error);
        alert('Error generating report. Please try again.');
    } finally {
        downloadReportBtn.innerHTML = originalText;
        downloadReportBtn.disabled = false;
    }
}

console.log('Product Defect Detection System - Frontend loaded');
