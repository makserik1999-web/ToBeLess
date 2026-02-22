// ToBeLess AI - Frontend JavaScript

// Utility: Show notification
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // Can be extended with toast notifications
}

// Utility: Format timestamp
function formatTimestamp(date) {
    return new Date(date).toLocaleString();
}

// Utility: Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    console.log('ToBeLess AI - Frontend initialized');
});
