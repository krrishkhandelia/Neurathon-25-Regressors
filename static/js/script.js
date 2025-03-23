document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadButton = document.getElementById('upload-button');
    const imagePreview = document.getElementById('image-preview');
    const filePreview = document.querySelector('.file-preview');
    const removePreview = document.getElementById('remove-preview');
    const filenameDisplay = document.querySelector('.filename-display');
    const scanTypeInputs = document.querySelectorAll('input[name="scan_type"]');
    
    if (!uploadArea || !fileInput) return;
    
    // Initialize form validation
    let fileSelected = false;
    let scanTypeSelected = false;
    
    // Handle drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight() {
        uploadArea.classList.remove('dragover');
    }
    
    // Handle file drop
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFileSelection(files);
        }
    }
    
    // Handle file selection via click
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFileSelection(this.files);
        }
    });
    
    function handleFileSelection(files) {
        const file = files[0];
        
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showError('Please select an image file (JPEG, PNG)');
            return;
        }
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('File is too large. Maximum file size is 10MB');
            return;
        }
        
        // Update preview
        const reader = new FileReader();
        
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            uploadArea.querySelector('.upload-content').classList.add('d-none');
            filePreview.classList.remove('d-none');
            filenameDisplay.textContent = file.name;
            
            fileSelected = true;
            checkFormValidity();
        };
        
        reader.readAsDataURL(file);
    }
    
    function clearFileInput() {
        fileInput.value = '';
        imagePreview.src = '';
        filePreview.classList.add('d-none');
        uploadArea.querySelector('.upload-content').classList.remove('d-none');
        filenameDisplay.textContent = '';
        
        fileSelected = false;
        checkFormValidity();
    }
    
    // Remove preview when close button is clicked
    if (removePreview) {
        removePreview.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent triggering upload area click
            clearFileInput();
        });
    }
    
    // Handle scan type selection
    scanTypeInputs.forEach(input => {
        input.addEventListener('change', function() {
            scanTypeSelected = true;
            checkFormValidity();
        });
    });
    
    function checkFormValidity() {
        if (fileSelected && scanTypeSelected) {
            uploadButton.disabled = false;
        } else {
            uploadButton.disabled = true;
        }
    }
    
    function showError(message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const formContainer = uploadArea.closest('.card-body');
        formContainer.insertBefore(alertDiv, formContainer.firstChild);
        
        // Automatically remove the alert after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
    
    // Result page zoom functionality
    const scanImage = document.querySelector('.scan-image');
    
    if (scanImage) {
        let scale = 1;
        let panning = false;
        let pointX = 0;
        let pointY = 0;
        let start = { x: 0, y: 0 };
        
        function setTransform() {
            scanImage.style.transform = `translate(${pointX}px, ${pointY}px) scale(${scale})`;
        }
        
        // Add zoom in and zoom out buttons if they exist
        const zoomIn = document.getElementById('zoom-in');
        const zoomOut = document.getElementById('zoom-out');
        const zoomReset = document.getElementById('zoom-reset');
        
        if (zoomIn) {
            zoomIn.addEventListener('click', function() {
                scale += 0.1;
                setTransform();
            });
        }
        
        if (zoomOut) {
            zoomOut.addEventListener('click', function() {
                if (scale > 0.5) {
                    scale -= 0.1;
                    setTransform();
                }
            });
        }
        
        if (zoomReset) {
            zoomReset.addEventListener('click', function() {
                scale = 1;
                pointX = 0;
                pointY = 0;
                setTransform();
            });
        }
    }
});