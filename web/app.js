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

function toggleDims() {
    const product = $("#product")?.value;
    const route = $("#route")?.value;

    // Toggle dimension sections
    $("#pipe-dims")?.classList.toggle("hidden", product !== "pipe");
    $("#sheet-dims")?.classList.toggle("hidden", product !== "sheet");

    // Toggle grade selection for conventional route
    const gradeWrap = $("#grade-wrap");
    if (gradeWrap) {
        gradeWrap.style.display = (route === "conventional") ? "block" : "none";
    }
}

function setupTabNavigation() {
    const tabButtons = $$(".tab-btn");
    const tabPanels = $$(".tab-panel");

    tabButtons.forEach(button => {
        button.addEventListener("click", () => {
            const targetTab = button.dataset.tab;

            // Update active states
            tabButtons.forEach(btn => btn.classList.remove("active"));
            tabPanels.forEach(panel => panel.classList.remove("active"));

            button.classList.add("active");
            const targetPanel = $(`#${targetTab}-panel`);
            if (targetPanel) {
                targetPanel.classList.add("active");
            }
        });
    });
}

function hideResultsSection() {
    const resultsSection = $("#results");
    if (resultsSection) {
        resultsSection.style.display = "none";
    }
}

function showResultsSection() {
    const resultsSection = $("#results");
    if (resultsSection) {
        resultsSection.style.display = "block";
        resultsSection.classList.add("fade-in");
    }
}
