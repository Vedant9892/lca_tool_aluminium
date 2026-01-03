// Enhanced LCA Dashboard JavaScript with FIXED Chart Sizing
const apiBase = "http://localhost:5000";

// Utility functions
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Global chart instances to manage updates
let chartInstances = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    toggleDims();
    setupTabNavigation();
    hideResultsSection();
}

function setupEventListeners() {
    $("#product")?.addEventListener("change", toggleDims);
    $("#route")?.addEventListener("change", toggleDims);
    $("#compute-btn")?.addEventListener("click", handleCompute);
    $("#table-search")?.addEventListener("input", handleTableSearch);
    $("#export-btn")?.addEventListener("click", exportTableData);
}