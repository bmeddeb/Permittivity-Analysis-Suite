/**
 * File Upload Drag and Drop Functionality
 * Handles drag-and-drop file uploads with validation and UI feedback
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeUploadHandlers();
});

function initializeUploadHandlers() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileSelectBtn = document.getElementById('file-select-btn');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const submitBtn = document.getElementById('submit-btn');
    
    // Only initialize if elements exist (on upload page)
    if (!dropZone || !fileInput) {
        return;
    }
    
    setupDragAndDropHandlers(dropZone);
    setupClickHandlers(dropZone, fileInput, fileSelectBtn);
    setupFileHandlers(fileInput, fileInfo, fileName, submitBtn, dropZone);
    setupRemoveHandler(removeFileBtn, fileInput, fileInfo, dropZone, submitBtn);
}

function setupDragAndDropHandlers(dropZone) {
    // Prevent default drag behaviors
    const dragEvents = ['dragenter', 'dragover', 'dragleave', 'drop'];
    
    dragEvents.forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => highlightDropZone(dropZone), false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => unhighlightDropZone(dropZone), false);
    });
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
}

function setupClickHandlers(dropZone, fileInput, fileSelectBtn) {
    // Handle click to browse files
    if (fileSelectBtn) {
        fileSelectBtn.addEventListener('click', () => fileInput.click());
    }
    
    // Make drop zone clickable
    dropZone.addEventListener('click', (e) => {
        if (e.target === dropZone || e.target.closest('.drop-zone-content')) {
            fileInput.click();
        }
    });
}

function setupFileHandlers(fileInput, fileInfo, fileName, submitBtn, dropZone) {
    // Handle file input change
    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFiles(files, fileInfo, fileName, submitBtn, dropZone, fileInput);
        }
    });
}

function setupRemoveHandler(removeFileBtn, fileInput, fileInfo, dropZone, submitBtn) {
    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', () => {
            removeFile(fileInput, fileInfo, dropZone, submitBtn);
        });
    }
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlightDropZone(dropZone) {
    dropZone.classList.add('drag-over');
}

function unhighlightDropZone(dropZone) {
    dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const submitBtn = document.getElementById('submit-btn');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    handleFiles(files, fileInfo, fileName, submitBtn, dropZone, fileInput);
}

function handleFiles(files, fileInfo, fileName, submitBtn, dropZone, fileInput) {
    if (files.length === 0) {
        return;
    }
    
    const file = files[0];
    
    // Validate file type
    if (!isValidCSVFile(file)) {
        showAlert('Please select a CSV file.', 'error');
        return;
    }
    
    // Validate file size (optional - 10MB limit)
    if (file.size > 10 * 1024 * 1024) {
        showAlert('File size must be less than 10MB.', 'error');
        return;
    }
    
    // Create a new FileList with the selected/dropped file
    try {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
    } catch (e) {
        // Fallback for older browsers
        console.warn('DataTransfer not supported, using fallback');
    }
    
    showFileInfo(file, fileInfo, fileName, submitBtn, dropZone);
}

function isValidCSVFile(file) {
    const validExtensions = ['.csv'];
    const fileName = file.name.toLowerCase();
    const validMimeTypes = ['text/csv', 'application/csv', 'text/plain'];
    
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
    const hasValidMimeType = validMimeTypes.includes(file.type) || file.type === '';
    
    return hasValidExtension && hasValidMimeType;
}

function showFileInfo(file, fileInfo, fileName, submitBtn, dropZone) {
    if (fileName) {
        fileName.textContent = file.name;
    }
    
    if (fileInfo) {
        fileInfo.classList.remove('hidden');
        fileInfo.classList.add('fade-in');
    }
    
    if (dropZone) {
        dropZone.classList.add('has-file');
    }
    
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.classList.remove('loading');
    }
}

function removeFile(fileInput, fileInfo, dropZone, submitBtn) {
    if (fileInput) {
        fileInput.value = '';
    }
    
    if (fileInfo) {
        fileInfo.classList.add('hidden');
        fileInfo.classList.remove('fade-in');
    }
    
    if (dropZone) {
        dropZone.classList.remove('has-file');
    }
    
    if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.classList.remove('loading');
    }
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    const bgColor = type === 'error' ? 'bg-red-100 border-red-400 text-red-700' : 
                   type === 'success' ? 'bg-green-100 border-green-400 text-green-700' :
                   'bg-blue-100 border-blue-400 text-blue-700';
    
    alertDiv.className = `mb-4 p-4 ${bgColor} rounded-lg border`;
    alertDiv.innerHTML = `
        <div class="flex justify-between items-center">
            <span>${message}</span>
            <button type="button" class="text-current hover:opacity-75" onclick="this.parentElement.parentElement.remove()">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                </svg>
            </button>
        </div>
    `;
    
    // Find container to insert alert
    const container = document.querySelector('main .container, .container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
}

// Form submission handler
function handleFormSubmission() {
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');
    
    if (form && submitBtn) {
        form.addEventListener('submit', function(e) {
            // Add loading state to submit button
            submitBtn.classList.add('loading');
            submitBtn.disabled = true;
            
            // Show progress (optional)
            showAlert('Uploading and processing your file...', 'info');
        });
    }
}

// Initialize form submission handler
document.addEventListener('DOMContentLoaded', handleFormSubmission);