// app.js - Fixed with wider borders and side-by-side layout

const API_BASE_URL = "http://127.0.0.1:8000";

// Tab switching function
function switchTab(tabName) {
    // Hide all tab content
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('nav a').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Hide intro section on tabs other than single
    document.querySelector('.project-intro').style.display = tabName === 'single' ? 'block' : 'none';
    
    // Clear previous results when switching tabs
    if (tabName === 'single') {
        clearSingleResults();
    } else if (tabName === 'batch') {
        clearBatchResults();
    }
}

function switchSpamTab(tabName) {
    // Hide all tab content
    document.querySelectorAll('.spam-tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.spam-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

function toggleFAQ(element) {
    const answer = element.nextElementSibling;
    const isActive = element.classList.contains('active');
    
    // Close all FAQs first
    document.querySelectorAll('.faq-question').forEach(q => {
        q.classList.remove('active');
    });
    document.querySelectorAll('.faq-answer').forEach(a => {
        a.classList.remove('active');
    });
    
    // Open clicked FAQ if not active
    if (!isActive) {
        element.classList.add('active');
        answer.classList.add('active');
    }
}

// Clear single prediction results
function clearSingleResults() {
    document.getElementById('resultText').textContent = 'Results will appear here...';
    document.getElementById('confidenceDisplay').innerHTML = '';
    document.getElementById('keywordsDisplay').innerHTML = '';
    document.getElementById('sms-input').value = '';
    document.getElementById('resultContainer').style.display = 'none';
}

// Clear batch upload results
function clearBatchResults() {
    const batchResult = document.getElementById('batchResult');
    if (batchResult) {
        batchResult.innerHTML = '';
        batchResult.style.display = 'none';
    }
    
    // Reset file input
    const fileInput = document.getElementById('csv-file');
    if (fileInput) fileInput.value = '';
    
    // Reset file label
    const fileLabel = document.querySelector('.file-label');
    const fileText = document.querySelector('.file-text');
    if (fileLabel && fileText) {
        fileText.textContent = 'Choose CSV file';
        fileLabel.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        fileLabel.style.background = 'rgba(255, 255, 255, 0.1)';
    }
}

// Single Prediction Form Handler
document.getElementById('single-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const smsInput = document.getElementById('sms-input').value.trim();
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
    const confidenceDisplay = document.getElementById('confidenceDisplay');
    const keywordsDisplay = document.getElementById('keywordsDisplay');
    
    if (!smsInput) {
        // Clear previous results when input is empty
        showResult('Please enter a message to analyze!', 'warning');
        confidenceDisplay.innerHTML = '';
        keywordsDisplay.innerHTML = '';
        return;
    }
    
    try {
        showResult('🔄 Analyzing message...', 'loading');
        confidenceDisplay.innerHTML = '';
        keywordsDisplay.innerHTML = '';
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: smsInput })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Get confidence level from API response
        const confidence = data.confidence || 0;
        
        // Determine confidence class based on percentage AND message type
        let confidenceClass = 'confidence-medium';
        if (data.label === 'spam') {
            // For spam messages - use red/orange colors
            if (confidence >= 80) confidenceClass = 'confidence-high-spam';
            else if (confidence >= 60) confidenceClass = 'confidence-medium-spam';
            else confidenceClass = 'confidence-low-spam';
        } else {
            // For ham messages - use green/blue colors
            if (confidence >= 80) confidenceClass = 'confidence-high-ham';
            else if (confidence >= 60) confidenceClass = 'confidence-medium-ham';
            else confidenceClass = 'confidence-low-ham';
        }
        
        // Display results based on prediction
        if (data.label === 'spam') {
            showResult(
                `SPAM DETECTED!`,
                'spam'
            );
            
            // Display confidence with dynamic styling
            confidenceDisplay.innerHTML = `
                <div class="result-confidence ${confidenceClass}">
                    Confidence: ${confidence}%
                </div>
            `;
            
            // Display spam keywords if available
            if (data.top_words && data.top_words.length > 0) {
                const keywordsHTML = data.top_words.map(word => 
                    `<span class="keyword-tag">${word[0]}</span>`
                ).join('');
                
                keywordsDisplay.innerHTML = `
                    <div class="spam-keywords">
                        <h4>Spam Keywords Detected:</h4>
                        <div class="keyword-tags">
                            ${keywordsHTML}
                        </div>
                    </div>
                `;
            }
            
        } else {
            showResult(
                `SAFE MESSAGE`,
                'ham'
            );
            
            // Display confidence with dynamic styling (now using ham colors)
            confidenceDisplay.innerHTML = `
                <div class="result-confidence ${confidenceClass}">
                    Confidence: ${confidence}%
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error:', error);
        showResult(
            `❌ Connection error to server!\n` +
            `Please check if backend is running?\n` +
            `Details: ${error.message}`,
            'error'
        );
        confidenceDisplay.innerHTML = '';
        keywordsDisplay.innerHTML = '';
    }
});

// Batch Upload Form Handler - KHUNG MỜ DÀI RA VÀ BẢNG 2 BÊN
document.getElementById('batch-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('csv-file');
    const batchResult = document.getElementById('batchResult');
    
    if (!fileInput.files[0]) {
        batchResult.innerHTML = '<div class="result-warning">Please select a CSV file!</div>';
        batchResult.style.display = 'block';
        return;
    }
    
    try {
        // Show loading state
        const submitBtn = document.querySelector('#batch-form .submit-btn');
        submitBtn.textContent = '🔄 Processing...';
        submitBtn.disabled = true;
        
        batchResult.innerHTML = '<div class="loading">🔄 Processing CSV file...</div>';
        batchResult.style.display = 'block';
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        const response = await fetch(`${API_BASE_URL}/batch-predict-json`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            // Tạo HTML cho 2 bảng với khung mờ dài ra
            let resultsHTML = `
                <div class="batch-results-container active" style="
                    background: rgba(255, 255, 255, 0.05); 
                    border-radius: 20px; 
                    padding: 30px; 
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    backdrop-filter: blur(10px);
                    margin-top: 25px;
                    width: 100%;
                    max-width: none;
                ">
                    <div class="batch-results-header" style="text-align: center; margin-bottom: 30px;">
                        <h3 class="batch-results-title" style="color: #ffd700; font-size: 1.8rem; margin-bottom: 10px;">Batch Analysis Results</h3>
                        <p class="batch-results-subtitle" style="color: rgba(255, 255, 255, 0.7);">Messages categorized as spam and legitimate</p>
                    </div>
                    
                    <div class="batch-stats" style="
                        display: grid;
                        grid-template-columns: repeat(3, 1fr);
                        gap: 20px;
                        margin-bottom: 40px;
                    ">
                        <div class="stat-card" style="
                            background: rgba(255, 255, 255, 0.08);
                            padding: 25px;
                            border-radius: 15px;
                            text-align: center;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                        ">
                            <div class="stat-value total-stat" style="color: #60a5fa; font-size: 2.5rem; font-weight: bold;">${data.total_messages}</div>
                            <div class="stat-label" style="color: rgba(255, 255, 255, 0.8);">Total Messages</div>
                        </div>
                        <div class="stat-card" style="
                            background: rgba(255, 255, 255, 0.08);
                            padding: 25px;
                            border-radius: 15px;
                            text-align: center;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                        ">
                            <div class="stat-value spam-stat" style="color: #ff6b6b; font-size: 2.5rem; font-weight: bold;">${data.spam_count}</div>
                            <div class="stat-label" style="color: rgba(255, 255, 255, 0.8);">Spam Detected</div>
                        </div>
                        <div class="stat-card" style="
                            background: rgba(255, 255, 255, 0.08);
                            padding: 25px;
                            border-radius: 15px;
                            text-align: center;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                        ">
                            <div class="stat-value ham-stat" style="color: #4ade80; font-size: 2.5rem; font-weight: bold;">${data.ham_count}</div>
                            <div class="stat-label" style="color: rgba(255, 255, 255, 0.8);">Legitimate</div>
                        </div>
                    </div>
                    
                    <div class="tables-container" style="
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 30px;
                        width: 100%;
                    ">
            `;
            
            // Separate spam and ham results
            const spamResults = data.results.filter(item => item.is_spam);
            const hamResults = data.results.filter(item => !item.is_spam);
            
            // Spam Table - KHUNG MỜ DÀI RA
            resultsHTML += `
                <div class="table-section spam-section" style="
                    border: 2px solid rgba(239, 68, 68, 0.6);
                    border-radius: 15px;
                    padding: 25px;
                    background: rgba(239, 68, 68, 0.08);
                    backdrop-filter: blur(5px);
                    width: 100%;
                    min-height: 400px;
                ">
                    <div class="section-header" style="
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        margin-bottom: 20px;
                        padding-bottom: 15px;
                        border-bottom: 1px solid rgba(239, 68, 68, 0.3);
                    ">
                        <span class="section-icon" style="font-size: 2rem;"></span>
                        <h4 class="section-title" style="color: #ef4444; font-size: 1.4rem; margin: 0; flex: 1;">Spam Messages</h4>
                        <span class="section-count" style="
                            background: rgba(239, 68, 68, 0.2);
                            color: #ef4444;
                            padding: 6px 12px;
                            border-radius: 15px;
                            font-size: 0.9rem;
                            font-weight: 600;
                        ">${spamResults.length} messages</span>
                    </div>
                    <table class="batch-results-table" style="
                        width: 100%;
                        border-collapse: collapse;
                        border: 1px solid rgba(239, 68, 68, 0.3);
                        border-radius: 10px;
                        overflow: hidden;
                    ">
                        <thead>
                            <tr>
                                <th style="
                                    background: rgba(239, 68, 68, 0.15);
                                    color: #ffd700;
                                    padding: 16px 20px;
                                    text-align: left;
                                    font-weight: 600;
                                    border-bottom: 2px solid #ef4444;
                                    font-size: 1rem;
                                ">Message Content</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            if (spamResults.length > 0) {
                spamResults.forEach(item => {
                    const displayMessage = item.text.length > 100 
                        ? item.text.substring(0, 100) + '...' 
                        : item.text;
                    
                    resultsHTML += `
                        <tr style="border-bottom: 1px solid rgba(239, 68, 68, 0.1);">
                            <td style="
                                padding: 14px 20px;
                                color: rgba(255, 255, 255, 0.9);
                                line-height: 1.5;
                                font-size: 0.95rem;
                            ">${displayMessage}</td>
                        </tr>
                    `;
                });
            } else {
                resultsHTML += `
                    <tr>
                        <td colspan="1" style="
                            text-align: center; 
                            padding: 40px; 
                            color: rgba(255, 255, 255, 0.6); 
                            font-style: italic;
                            font-size: 1rem;
                        ">No spam messages detected</td>
                    </tr>
                `;
            }
            
            resultsHTML += `
                        </tbody>
                    </table>
                </div>
            `;
            
            // Ham Table - KHUNG MỜ DÀI RA
            resultsHTML += `
                <div class="table-section ham-section" style="
                    border: 2px solid rgba(74, 222, 128, 0.6);
                    border-radius: 15px;
                    padding: 25px;
                    background: rgba(74, 222, 128, 0.08);
                    backdrop-filter: blur(5px);
                    width: 100%;
                    min-height: 400px;
                ">
                    <div class="section-header" style="
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        margin-bottom: 20px;
                        padding-bottom: 15px;
                        border-bottom: 1px solid rgba(74, 222, 128, 0.3);
                    ">
                        <span class="section-icon" style="font-size: 2rem;"></span>
                        <h4 class="section-title" style="color: #4ade80; font-size: 1.4rem; margin: 0; flex: 1;">Legitimate Messages</h4>
                        <span class="section-count" style="
                            background: rgba(74, 222, 128, 0.2);
                            color: #4ade80;
                            padding: 6px 12px;
                            border-radius: 15px;
                            font-size: 0.9rem;
                            font-weight: 600;
                        ">${hamResults.length} messages</span>
                    </div>
                    <table class="batch-results-table" style="
                        width: 100%;
                        border-collapse: collapse;
                        border: 1px solid rgba(74, 222, 128, 0.3);
                        border-radius: 10px;
                        overflow: hidden;
                    ">
                        <thead>
                            <tr>
                                <th style="
                                    background: rgba(74, 222, 128, 0.15);
                                    color: #ffd700;
                                    padding: 16px 20px;
                                    text-align: left;
                                    font-weight: 600;
                                    border-bottom: 2px solid #4ade80;
                                    font-size: 1rem;
                                ">Message Content</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            if (hamResults.length > 0) {
                hamResults.forEach(item => {
                    const displayMessage = item.text.length > 100 
                        ? item.text.substring(0, 100) + '...' 
                        : item.text;
                    
                    resultsHTML += `
                        <tr style="border-bottom: 1px solid rgba(74, 222, 128, 0.1);">
                            <td style="
                                padding: 14px 20px;
                                color: rgba(255, 255, 255, 0.9);
                                line-height: 1.5;
                                font-size: 0.95rem;
                            ">${displayMessage}</td>
                        </tr>
                    `;
                });
            } else {
                resultsHTML += `
                    <tr>
                        <td colspan="1" style="
                            text-align: center; 
                            padding: 40px; 
                            color: rgba(255, 255, 255, 0.6); 
                            font-style: italic;
                            font-size: 1rem;
                        ">No legitimate messages found</td>
                    </tr>
                `;
            }
            
            resultsHTML += `
                        </tbody>
                    </table>
                </div>
            `;
            
            // Close containers
            resultsHTML += `
                    </div>
                </div>
            `;
            
            batchResult.innerHTML = resultsHTML;
            
        } else {
            throw new Error(data.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('Batch upload error:', error);
        batchResult.innerHTML = `
            <div class="result-error">
                ❌ Batch processing error!\n${error.message}
            </div>
        `;
    } finally {
        // Reset button state
        const submitBtn = document.querySelector('#batch-form .submit-btn');
        submitBtn.textContent = 'Process Batch';
        submitBtn.disabled = false;
    }
});

// File input change handler
document.getElementById('csv-file').addEventListener('change', function(e) {
    const fileLabel = document.querySelector('.file-label');
    const fileText = document.querySelector('.file-text');
    
    if (this.files[0]) {
        fileText.textContent = this.files[0].name;
        fileLabel.style.borderColor = '#4ade80';
        fileLabel.style.background = 'rgba(74, 222, 128, 0.1)';
    } else {
        fileText.textContent = 'Choose CSV file';
        fileLabel.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        fileLabel.style.background = 'rgba(255, 255, 255, 0.1)';
    }
});

// Show result function
function showResult(message, type) {
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
    
    // Reset classes
    resultContainer.className = 'result-container';
    resultText.className = '';
    
    // Apply type-specific styling
    switch (type) {
        case 'spam':
            resultContainer.classList.add('result-spam');
            resultText.classList.add('result-spam');
            break;
        case 'ham':
            resultContainer.classList.add('result-ham');
            resultText.classList.add('result-ham');
            break;
        case 'error':
            resultContainer.classList.add('result-error');
            resultText.classList.add('result-error');
            break;
        case 'warning':
            resultContainer.classList.add('result-warning');
            resultText.classList.add('result-warning');
            break;
        case 'loading':
            resultContainer.classList.add('loading');
            resultText.classList.add('loading');
            break;
    }
    
    resultText.textContent = message;
    resultContainer.style.display = 'block';
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 AmongSMS Frontend Initialized');
    
    // Test backend connection on load
    testBackendConnection();
    
    // Auto-open first FAQ when entering Other Info tab
    const firstFAQ = document.querySelector('.faq-question');
    if (firstFAQ && window.location.hash !== '#single' && window.location.hash !== '#batch') {
        firstFAQ.classList.add('active');
        firstFAQ.nextElementSibling.classList.add('active');
    }
});

// Test backend connection
async function testBackendConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend connected:', data);
        } else {
            console.warn('⚠️ Backend health check failed');
        }
    } catch (error) {
        console.error('❌ Cannot connect to backend:', error);
    }
}