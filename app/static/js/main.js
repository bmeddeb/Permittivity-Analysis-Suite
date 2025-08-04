/**
 * Permittivity Analysis Suite - Main JavaScript
 * Global utilities and common functionality
 */

// Global app namespace
window.PermittivityApp = window.PermittivityApp || {};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    PermittivityApp.init();
});

PermittivityApp.init = function() {
    this.initializeTooltips();
    this.initializeAlerts();
    this.initializeAnimations();
    this.setupGlobalEventListeners();
};

/**
 * Initialize tooltips (now using custom implementation since no Bootstrap)
 */
PermittivityApp.initializeTooltips = function() {
    // Custom tooltip implementation for Tailwind
    const tooltipTriggerList = document.querySelectorAll('[title]');
    tooltipTriggerList.forEach(function(el) {
        // Basic tooltip functionality can be added here if needed
    });
};

/**
 * Auto-dismiss alerts after a delay
 */
PermittivityApp.initializeAlerts = function() {
    const alerts = document.querySelectorAll('.alert-auto-dismiss');
    alerts.forEach(function(alert) {
        // Auto-dismiss alerts after 5 seconds
        setTimeout(function() {
            if (alert.parentNode) {
                alert.remove();
            }
        }, 5000);
    });
};

/**
 * Add fade-in animations to cards and content
 */
PermittivityApp.initializeAnimations = function() {
    const cards = document.querySelectorAll('.bg-white');
    cards.forEach(function(card, index) {
        setTimeout(function() {
            card.classList.add('fade-in');
        }, index * 100);
    });
};

/**
 * Setup global event listeners
 */
PermittivityApp.setupGlobalEventListeners = function() {
    // Add loading state to buttons with data-loading attribute
    const loadingButtons = document.querySelectorAll('[data-loading]');
    loadingButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            this.classList.add('loading');
            this.disabled = true;
            
            // Re-enable after 5 seconds as fallback
            setTimeout(() => {
                this.classList.remove('loading');
                this.disabled = false;
            }, 5000);
        });
    });
    
    // Smooth scroll for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    anchorLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
};

/**
 * Utility function to show dynamic alerts
 */
PermittivityApp.showAlert = function(message, type = 'info', duration = 5000) {
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
    
    // Insert at the top of the main content area
    const container = document.querySelector('main .container, .container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after specified duration
        if (duration > 0) {
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, duration);
        }
    }
    
    return alertDiv;
};

/**
 * Utility function to format file sizes
 */
PermittivityApp.formatFileSize = function(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Utility function to format numbers
 */
PermittivityApp.formatNumber = function(number, decimals = 3) {
    return parseFloat(number).toFixed(decimals);
};

/**
 * Utility function to validate email
 */
PermittivityApp.validateEmail = function(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
};

/**
 * Utility function to debounce function calls
 */
PermittivityApp.debounce = function(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        
        if (callNow) func.apply(context, args);
    };
};

/**
 * Local storage helpers
 */
PermittivityApp.storage = {
    set: function(key, value) {
        try {
            localStorage.setItem('permittivity_' + key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.warn('Failed to save to localStorage:', e);
            return false;
        }
    },
    
    get: function(key, defaultValue = null) {
        try {
            const item = localStorage.getItem('permittivity_' + key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.warn('Failed to read from localStorage:', e);
            return defaultValue;
        }
    },
    
    remove: function(key) {
        try {
            localStorage.removeItem('permittivity_' + key);
            return true;
        } catch (e) {
            console.warn('Failed to remove from localStorage:', e);
            return false;
        }
    }
};