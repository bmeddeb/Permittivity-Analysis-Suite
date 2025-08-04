/**
 * Trial Management - Editable Table and CRUD Operations
 * Handles inline editing, deletion, and trial management
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeTrialManagement();
});

function initializeTrialManagement() {
    const dataTable = document.getElementById('material-data-table');
    const editButton = document.getElementById('edit-mode-btn');
    const saveButton = document.getElementById('save-mode-btn');
    const cancelButton = document.getElementById('cancel-mode-btn');
    const addRowButton = document.getElementById('add-row-btn');
    const deleteTrialButton = document.getElementById('delete-trial-btn');
    const editTrialButton = document.getElementById('edit-trial-btn');
    
    // Bottom buttons
    const saveButtonBottom = document.getElementById('save-mode-btn-bottom');
    const cancelButtonBottom = document.getElementById('cancel-mode-btn-bottom');
    const addRowButtonBottom = document.getElementById('add-row-btn-bottom');
    const bottomControls = document.getElementById('bottom-controls');
    
    if (!dataTable) return; // Not on trial page
    
    let isEditMode = false;
    let originalData = new Map(); // Store original values for cancel functionality
    let newRowCounter = 0; // Counter for temporary IDs of new rows
    
    // Initialize event listeners
    if (editButton) {
        editButton.addEventListener('click', () => toggleEditMode(true));
    }
    
    if (saveButton) {
        saveButton.addEventListener('click', saveAllChanges);
    }
    
    if (cancelButton) {
        cancelButton.addEventListener('click', () => toggleEditMode(false));
    }
    
    if (addRowButton) {
        addRowButton.addEventListener('click', addNewRow);
    }
    
    // Bottom button event listeners
    if (saveButtonBottom) {
        saveButtonBottom.addEventListener('click', saveAllChanges);
    }
    
    if (cancelButtonBottom) {
        cancelButtonBottom.addEventListener('click', () => toggleEditMode(false));
    }
    
    if (addRowButtonBottom) {
        addRowButtonBottom.addEventListener('click', addNewRow);
    }
    
    if (deleteTrialButton) {
        deleteTrialButton.addEventListener('click', deleteTrialConfirm);
    }
    
    if (editTrialButton) {
        editTrialButton.addEventListener('click', editTrialName);
    }
    
    function toggleEditMode(enable) {
        isEditMode = enable;
        const rows = dataTable.querySelectorAll('tbody tr');
        
        if (enable) {
            // Store original values
            originalData.clear();
            rows.forEach(row => {
                const dataId = row.dataset.dataId;
                const cells = row.querySelectorAll('td');
                originalData.set(dataId, {
                    frequency: cells[0].textContent.trim(),
                    dk: cells[1].textContent.trim(),
                    df: cells[2].textContent.trim()
                });
                
                // Make cells editable
                makeRowEditable(row);
                
                // Add delete button
                addDeleteButton(row);
            });
            
            // Show/hide top buttons
            editButton.classList.add('hidden');
            saveButton.classList.remove('hidden');
            cancelButton.classList.remove('hidden');
            if (addRowButton) addRowButton.classList.remove('hidden');
            
            // Show/hide bottom buttons
            if (bottomControls) bottomControls.style.display = 'flex';
            if (saveButtonBottom) saveButtonBottom.classList.remove('hidden');
            if (cancelButtonBottom) cancelButtonBottom.classList.remove('hidden');
            if (addRowButtonBottom) addRowButtonBottom.classList.remove('hidden');
            
        } else {
            // Restore original values and disable editing
            rows.forEach(row => {
                const dataId = row.dataset.dataId;
                const originalValues = originalData.get(dataId);
                if (originalValues) {
                    const cells = row.querySelectorAll('td');
                    cells[0].innerHTML = originalValues.frequency;
                    cells[1].innerHTML = originalValues.dk;
                    cells[2].innerHTML = originalValues.df;
                }
                
                makeRowReadOnly(row);
                removeDeleteButton(row);
            });
            
            // Show/hide top buttons
            editButton.classList.remove('hidden');
            saveButton.classList.add('hidden');
            cancelButton.classList.add('hidden');
            if (addRowButton) addRowButton.classList.add('hidden');
            
            // Show/hide bottom buttons
            if (bottomControls) bottomControls.style.display = 'none';
            if (saveButtonBottom) saveButtonBottom.classList.add('hidden');
            if (cancelButtonBottom) cancelButtonBottom.classList.add('hidden');
            if (addRowButtonBottom) addRowButtonBottom.classList.add('hidden');
            
            // Remove any unsaved new rows
            const newRows = dataTable.querySelectorAll('tbody tr[data-is-new="true"]');
            newRows.forEach(row => row.remove());
            
            originalData.clear();
        }
    }
    
    function makeRowEditable(row) {
        const cells = row.querySelectorAll('td');
        
        // Make first 3 cells editable (frequency, dk, df)
        for (let i = 0; i < 3; i++) {
            const cell = cells[i];
            const value = cell.textContent.trim();
            const input = document.createElement('input');
            input.type = 'number';
            input.step = i === 2 ? '0.000001' : '0.001'; // More precision for df
            input.value = value;
            input.className = 'w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-blue-500 focus:border-blue-500';
            input.style.minWidth = '100px';
            
            // Add validation
            input.addEventListener('input', validateInput);
            
            cell.innerHTML = '';
            cell.appendChild(input);
        }
        
        row.classList.add('editable-row');
    }
    
    function makeRowReadOnly(row) {
        const cells = row.querySelectorAll('td');
        
        // Convert inputs back to text
        for (let i = 0; i < 3; i++) {
            const cell = cells[i];
            const input = cell.querySelector('input');
            if (input) {
                const value = parseFloat(input.value);
                const decimals = i === 2 ? 6 : 3; // More decimals for df
                cell.innerHTML = value.toFixed(decimals);
            }
        }
        
        row.classList.remove('editable-row');
    }
    
    function addDeleteButton(row) {
        const actionsCell = row.querySelector('td:last-child');
        if (!actionsCell) {
            // Add actions cell if it doesn't exist
            const newCell = document.createElement('td');
            row.appendChild(newCell);
        }
        
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'text-red-500 hover:text-red-700 p-1 rounded transition-colors';
        deleteBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>';
        deleteBtn.title = 'Delete this data point';
        deleteBtn.addEventListener('click', () => deleteDataPoint(row));
        
        const finalCell = row.querySelector('td:last-child');
        finalCell.innerHTML = '';
        finalCell.appendChild(deleteBtn);
    }
    
    function removeDeleteButton(row) {
        const actionsCell = row.querySelector('td:last-child');
        if (actionsCell) {
            actionsCell.innerHTML = '';
        }
    }
    
    function validateInput(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        
        input.classList.remove('is-invalid', 'is-valid');
        
        if (isNaN(value) || value <= 0) {
            input.classList.add('is-invalid');
        } else {
            input.classList.add('is-valid');
        }
    }
    
    async function saveAllChanges() {
        const rows = dataTable.querySelectorAll('tbody tr.editable-row');
        const updates = [];
        const newRows = [];
        let hasErrors = false;
        
        // Collect all changes and new rows
        rows.forEach(row => {
            const dataId = row.dataset.dataId;
            const inputs = row.querySelectorAll('input');
            const isNew = row.dataset.isNew === 'true';
            
            // Validate inputs
            for (let input of inputs) {
                if (input.classList.contains('is-invalid') || !input.value.trim()) {
                    hasErrors = true;
                    input.focus();
                }
            }
            
            if (!hasErrors && inputs.length >= 3) {
                const data = {
                    frequency_ghz: parseFloat(inputs[0].value),
                    dk: parseFloat(inputs[1].value),
                    df: parseFloat(inputs[2].value)
                };
                
                if (isNew) {
                    newRows.push(data);
                } else {
                    updates.push({
                        id: dataId,
                        ...data
                    });
                }
            }
        });
        
        if (hasErrors) {
            PermittivityApp.showAlert('Please fix all validation errors before saving.', 'danger');
            return;
        }
        
        if (updates.length === 0 && newRows.length === 0) {
            toggleEditMode(false);
            return;
        }
        
        // Show loading state on both save buttons
        const saveButtons = [saveButton, saveButtonBottom].filter(btn => btn);
        saveButtons.forEach(btn => {
            btn.disabled = true;
            btn.innerHTML = '<i class="bi bi-hourglass-split"></i> Saving...';
        });
        
        try {
            // Save updates
            for (let update of updates) {
                await updateMaterialData(update);
            }
            
            // Save new rows
            const trialId = window.location.pathname.split('/').pop();
            for (let newData of newRows) {
                await addMaterialData(trialId, newData);
            }
            
            const totalChanges = updates.length + newRows.length;
            PermittivityApp.showAlert(`Successfully saved ${totalChanges} changes!`, 'success');
            toggleEditMode(false);
            
            // Refresh the page to show updated data and summary
            setTimeout(() => {
                location.reload();
            }, 1000);
            
        } catch (error) {
            PermittivityApp.showAlert(`Error saving changes: ${error.message}`, 'danger');
        } finally {
            // Restore save buttons
            saveButtons.forEach(btn => {
                btn.disabled = false;
                btn.innerHTML = '<i class="bi bi-check-lg"></i> Save Changes';
            });
        }
    }
    
    async function updateMaterialData(data) {
        const response = await fetch(`/api/material-data/${data.id}/update`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.message);
        }
        
        return result;
    }
    
    async function addMaterialData(trialId, data) {
        const response = await fetch(`/api/trial/${trialId}/add-data`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.message);
        }
        
        return result;
    }
    
    async function deleteDataPoint(row) {
        const dataId = row.dataset.dataId;
        
        if (!confirm('Are you sure you want to delete this data point? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/material-data/${dataId}/delete`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Remove row from table
                row.remove();
                
                PermittivityApp.showAlert(result.message, 'success');
                
                // If trial was deleted, redirect to home
                if (result.trial_deleted) {
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 2000);
                }
                
                // Update summary statistics
                updateSummaryStats();
                
            } else {
                PermittivityApp.showAlert(result.message, 'danger');
            }
            
        } catch (error) {
            PermittivityApp.showAlert(`Error deleting data point: ${error.message}`, 'danger');
        }
    }
    
    function updateSummaryStats() {
        const rows = dataTable.querySelectorAll('tbody tr');
        const dataPointsCount = document.querySelector('[data-stat="data-points"]');
        
        if (dataPointsCount) {
            dataPointsCount.textContent = rows.length;
        }
        
        // Recalculate ranges if there are still rows
        if (rows.length > 0) {
            const frequencies = [];
            const dks = [];
            const dfs = [];
            
            rows.forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length >= 3) {
                    frequencies.push(parseFloat(cells[0].textContent));
                    dks.push(parseFloat(cells[1].textContent));
                    dfs.push(parseFloat(cells[2].textContent));
                }
            });
            
            // Update frequency range
            const freqRange = document.querySelector('[data-stat="freq-range"]');
            if (freqRange && frequencies.length > 0) {
                const min = Math.min(...frequencies).toFixed(3);
                const max = Math.max(...frequencies).toFixed(3);
                freqRange.textContent = `${min} - ${max} GHz`;
            }
            
            // Update Dk range
            const dkRange = document.querySelector('[data-stat="dk-range"]');
            if (dkRange && dks.length > 0) {
                const min = Math.min(...dks).toFixed(3);
                const max = Math.max(...dks).toFixed(3);
                dkRange.textContent = `${min} - ${max}`;
            }
            
            // Update Df range
            const dfRange = document.querySelector('[data-stat="df-range"]');
            if (dfRange && dfs.length > 0) {
                const min = Math.min(...dfs).toFixed(6);
                const max = Math.max(...dfs).toFixed(6);
                dfRange.textContent = `${min} - ${max}`;
            }
        }
    }
}

function deleteTrialConfirm() {
    const trialName = document.querySelector('[data-trial-name]')?.dataset.trialName || 'this trial';
    
    if (confirm(`Are you sure you want to delete "${trialName}"? This will permanently delete all associated data and cannot be undone.`)) {
        // Submit the delete form
        const deleteForm = document.getElementById('delete-trial-form');
        if (deleteForm) {
            deleteForm.submit();
        }
    }
}

function editTrialName() {
    const trialNameElement = document.querySelector('[data-trial-name]');
    const currentName = trialNameElement.textContent.trim();
    
    const newName = prompt('Enter new trial name:', currentName);
    
    if (newName && newName.trim() !== currentName) {
        updateTrialName(newName.trim());
    }
}

async function updateTrialName(newName) {
    const trialId = window.location.pathname.split('/').pop();
    
    try {
        const formData = new FormData();
        formData.append('filename', newName);
        
        const response = await fetch(`/trial/${trialId}/update`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update the displayed name
            const trialNameElement = document.querySelector('[data-trial-name]');
            trialNameElement.textContent = newName;
            
            // Update page title
            document.title = `Trial: ${newName} - Permittivity Analysis Suite`;
            
            PermittivityApp.showAlert(result.message, 'success');
        } else {
            PermittivityApp.showAlert(result.message, 'danger');
        }
        
    } catch (error) {
        PermittivityApp.showAlert(`Error updating trial name: ${error.message}`, 'danger');
    }
}

// Add new row functionality
function addNewRow() {
    const dataTable = document.getElementById('material-data-table');
    if (!dataTable) return;
    
    const newRowCounter = Date.now(); // Use timestamp as unique ID
    const tempId = `new_${newRowCounter}`;
    
    // Create new row element
    const newRow = document.createElement('tr');
    newRow.dataset.dataId = tempId;
    newRow.dataset.isNew = 'true';
    newRow.classList.add('editable-row', 'new-row');
    
    // Add cells with input fields
    const frequencyCell = document.createElement('td');
    const dkCell = document.createElement('td');
    const dfCell = document.createElement('td');
    const actionCell = document.createElement('td');
    
    // Create input fields
    const frequencyInput = createInputField('number', '', '0.001');
    const dkInput = createInputField('number', '', '0.001');
    const dfInput = createInputField('number', '', '0.000001');
    
    frequencyInput.placeholder = 'Frequency (GHz)';
    dkInput.placeholder = 'Dk';
    dfInput.placeholder = 'Df';
    
    // Add validation
    frequencyInput.addEventListener('input', validateInput);
    dkInput.addEventListener('input', validateInput);
    dfInput.addEventListener('input', validateInput);
    
    frequencyCell.appendChild(frequencyInput);
    dkCell.appendChild(dkInput);
    dfCell.appendChild(dfInput);
    
    // Add delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'text-red-500 hover:text-red-700 p-1 rounded transition-colors';
    deleteBtn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>';
    deleteBtn.title = 'Delete this data point';
    deleteBtn.addEventListener('click', () => {
        newRow.remove();
        updateSummaryStatsFromTable();
    });
    
    actionCell.appendChild(deleteBtn);
    
    // Append cells to row
    newRow.appendChild(frequencyCell);
    newRow.appendChild(dkCell);
    newRow.appendChild(dfCell);
    newRow.appendChild(actionCell);
    
    // Insert row at the end of tbody
    const tbody = dataTable.querySelector('tbody');
    tbody.appendChild(newRow);
    
    // Focus on the first input
    frequencyInput.focus();
    
    // Update summary stats
    updateSummaryStatsFromTable();
}

function createInputField(type, value, step) {
    const input = document.createElement('input');
    input.type = type;
    input.step = step;
    input.value = value;
    input.className = 'w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-blue-500 focus:border-blue-500';
    input.style.minWidth = '100px';
    return input;
}

function validateInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);
    
    input.classList.remove('border-red-500', 'bg-red-50', 'border-green-500', 'bg-green-50');
    
    if (isNaN(value) || value < 0) {
        input.classList.add('border-red-500', 'bg-red-50');
    } else if (value > 0) {
        input.classList.add('border-green-500', 'bg-green-50');
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

function updateSummaryStatsFromTable() {
    const dataTable = document.getElementById('material-data-table');
    if (!dataTable) return;
    
    const rows = dataTable.querySelectorAll('tbody tr');
    const dataPointsCount = document.querySelector('[data-stat="data-points"]');
    
    if (dataPointsCount) {
        dataPointsCount.textContent = rows.length;
    }
    
    // Recalculate ranges if there are still rows
    if (rows.length > 0) {
        const frequencies = [];
        const dks = [];
        const dfs = [];
        
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 3) {
                // Handle both text content and input values
                let freqValue, dkValue, dfValue;
                
                const freqInput = cells[0].querySelector('input');
                const dkInput = cells[1].querySelector('input');
                const dfInput = cells[2].querySelector('input');
                
                if (freqInput) {
                    freqValue = parseFloat(freqInput.value);
                    dkValue = parseFloat(dkInput.value);
                    dfValue = parseFloat(dfInput.value);
                } else {
                    freqValue = parseFloat(cells[0].textContent);
                    dkValue = parseFloat(cells[1].textContent);
                    dfValue = parseFloat(cells[2].textContent);
                }
                
                if (!isNaN(freqValue)) frequencies.push(freqValue);
                if (!isNaN(dkValue)) dks.push(dkValue);
                if (!isNaN(dfValue)) dfs.push(dfValue);
            }
        });
        
        // Update frequency range
        const freqRange = document.querySelector('[data-stat="freq-range"]');
        if (freqRange && frequencies.length > 0) {
            const min = Math.min(...frequencies).toFixed(3);
            const max = Math.max(...frequencies).toFixed(3);
            freqRange.textContent = `${min} - ${max} GHz`;
        }
        
        // Update Dk range
        const dkRange = document.querySelector('[data-stat="dk-range"]');
        if (dkRange && dks.length > 0) {
            const min = Math.min(...dks).toFixed(3);
            const max = Math.max(...dks).toFixed(3);
            dkRange.textContent = `${min} - ${max}`;
        }
        
        // Update Df range
        const dfRange = document.querySelector('[data-stat="df-range"]');
        if (dfRange && dfs.length > 0) {
            const min = Math.min(...dfs).toFixed(6);
            const max = Math.max(...dfs).toFixed(6);
            dfRange.textContent = `${min} - ${max}`;
        }
    }
}