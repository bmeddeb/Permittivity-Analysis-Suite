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
 * Initialize Bootstrap tooltips
 */
PermittivityApp.initializeTooltips = function() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
};

/**
 * Auto-dismiss alerts after a delay
 */
PermittivityApp.initializeAlerts = function() {
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(function(alert) {
        // Auto-dismiss success and info alerts after 5 seconds
        if (alert.classList.contains('alert-success') || alert.classList.contains('alert-info')) {
            setTimeout(function() {
                const bsAlert = new bootstrap.Alert(alert);
                if (alert.parentNode) {
                    bsAlert.close();
                }
            }, 5000);
        }
    });
};

/**
 * Add fade-in animations to cards and content
 */
PermittivityApp.initializeAnimations = function() {
    const cards = document.querySelectorAll('.card');
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
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main content area
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remove after specified duration
        if (duration > 0) {
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    const bsAlert = new bootstrap.Alert(alertDiv);
                    bsAlert.close();
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