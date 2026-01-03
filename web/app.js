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

async function handleCompute() {
    const button = $("#compute-btn");
    const status = $("#status");
    if (!button || !status) return;

    // Update UI state
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Calculating...';
    status.textContent = "Processing environmental impact assessment...";

    try {
        const inputData = collectInputData();

        // --- ADD VALIDATION HERE ---
        if (
            inputData.product === "pipe" &&
            inputData.inner_radius_m != null &&
            inputData.outer_radius_m != null &&
            inputData.inner_radius_m > inputData.outer_radius_m
        ) {
            throw new Error("Inner radius cannot be greater than outer radius.");
        }

        console.log("Sending data:", inputData); // Debug log

        const response = await fetch(`${apiBase}/compute`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        console.log("Received data:", data); // Debug log

        // Update UI with results
        updateSummaryCards(data.totals.total);
        updateChartsAndVisualizations(data);
        updateResultsTable(data.stages);

        // Show results section
        showResultsSection();

        status.textContent = "Calculation completed successfully";
    } catch (error) {
        console.error("Error during computation:", error);
        status.textContent = `Error: ${error.message}`;
        hideResultsSection();
    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-play"></i> Calculate Impact';
    }
}


function collectInputData() {
    const product = $("#product")?.value || "pipe";
    const inputData = {
        product: product,
        units: parseInt($("#units")?.value) || 100,
        route_type: $("#route")?.value || "conventional",
        bauxite_grade: $("#grade")?.value || "medium",
        energy_source: $("#energy")?.value || "grid_mix",
        eol_option: $("#eol")?.value || "recycle"
    };

    // Add dimensions based on product type with correct parameter names
    if (product === "pipe") {
        inputData.outer_radius_m = parseFloat($("#outer_radius")?.value) || 0.1;
        inputData.inner_radius_m = parseFloat($("#inner_radius")?.value) || 0.08;
        inputData.length_m = parseFloat($("#pipe_length")?.value) || 2.0;
        // Set sheet dimensions to null for pipe
        inputData.sheet_length_m = null;
        inputData.width_m = null;
        inputData.thickness_m = null;
    } else {
        inputData.sheet_length_m = parseFloat($("#sheet_length")?.value) || 2.0;
        inputData.width_m = parseFloat($("#width")?.value) || 1.0;
        inputData.thickness_m = parseFloat($("#thickness")?.value) || 0.005;
        // Set pipe dimensions to null for sheet
        inputData.outer_radius_m = null;
        inputData.inner_radius_m = null;
        inputData.length_m = null;
    }

    return inputData;
}

function updateSummaryCards(totals) {
    const summaryGrid = $("#summary-cards");
    if (!summaryGrid || !totals) return;

    const cards = [
        {
            title: "Total CO₂",
            value: totals.carbon_kgco2e?.toFixed(1) || "0.0",
            unit: "kg",
            icon: "fas fa-leaf",
            class: "danger"
        },
        {
            title: "Total Electricity",
            value: totals.electricity_kwh?.toFixed(0) || "0",
            unit: "kWh",
            icon: "fas fa-bolt",
            class: "warning"
        },
        {
            title: "Total Wastewater",
            value: totals.wastewater_l?.toFixed(0) || "0",
            unit: "L",
            icon: "fas fa-tint",
            class: "info"
        },
        {
            title: "Total Cost",
            value: "$" + ((totals.manufacturing_cost_per_unit || 0) + (totals.transport_cost_usd || 0)).toFixed(0),
            unit: "USD",
            icon: "fas fa-dollar-sign",
            class: "success"
        }
    ];

    summaryGrid.innerHTML = cards.map(card => `
        <div class="summary-card ${card.class}">
            <div class="icon ${card.icon}"></div>
            <h4>${card.title}</h4>
            <div class="value">
                ${card.value}
                <span class="unit">${card.unit}</span>
            </div>
        </div>
    `).join("");
}

function updateChartsAndVisualizations(data) {
    if (!data.stages) return;

    // Filter data for total scope
    const totalData = data.stages.filter(stage => stage.scope === "total");

    if (totalData.length === 0) {
        console.warn("No total scope data found");
        return;
    }

    // Update comparison charts
    updateComparisonCharts(totalData);

    // Update distribution charts  
    updateDistributionCharts(totalData);

    // Update process flow chart
    updateProcessFlowChart(totalData);

    // Update efficiency charts
    updateEfficiencyCharts(totalData);
}

// Base chart options with FIXED sizing
function getBaseChartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false, // CRITICAL: Allow custom sizing
        aspectRatio: false, // Disable aspect ratio
        plugins: {
            legend: { 
                display: false,
                labels: { color: "#b4c6fc" }
            },
            title: { display: false }
        },
        layout: {
            padding: {
                top: 10,
                bottom: 10,
                left: 10,
                right: 10
            }
        }
    };
}
