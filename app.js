// app.js - Fixed version
const API_BASE_URL = "http://localhost:8000";

// ==================== TAB MANAGEMENT FUNCTIONS ====================

/**
 * Chuyển đổi giữa các tab chính (single/batch)
 * @param {string} tabName - Tên tab cần hiển thị ('single' hoặc 'batch')
 */
function switchTab(tabName) {
    // Ẩn tất cả nội dung tab
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Xóa class active từ tất cả button
    document.querySelectorAll('nav a').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Hiển thị tab được chọn
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Thêm class active cho button được click
    event.target.classList.add('active');
    
    // Ẩn section intro trên các tab khác 'single'
    document.querySelector('.project-intro').style.display = tabName === 'single' ? 'block' : 'none';
    
    // Xóa kết quả trước đó khi chuyển tab
    if (tabName === 'single') {
        clearSingleResults();
    } else if (tabName === 'batch') {
        clearBatchResults();
    }
}

/**
 * Chuyển đổi giữa các tab spam (dành cho phần hiển thị kết quả)
 * @param {string} tabName - Tên tab spam cần hiển thị
 */
function switchSpamTab(tabName) {
    // Ẩn tất cả nội dung tab spam
    document.querySelectorAll('.spam-tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Xóa class active từ tất cả button spam
    document.querySelectorAll('.spam-tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Hiển thị tab spam được chọn
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Thêm class active cho button spam được click
    event.target.classList.add('active');
}

// ==================== FAQ FUNCTIONS ====================

/**
 * Mở/đóng FAQ items
 * @param {Element} element - Phần tử FAQ question được click
 */
function toggleFAQ(element) {
    const answer = element.nextElementSibling;
    const isActive = element.classList.contains('active');
    
    // Đóng tất cả FAQs trước
    document.querySelectorAll('.faq-question').forEach(q => {
        q.classList.remove('active');
    });
    document.querySelectorAll('.faq-answer').forEach(a => {
        a.classList.remove('active');
    });
    
    // Mở FAQ được click nếu chưa active
    if (!isActive) {
        element.classList.add('active');
        answer.classList.add('active');
    }
}

// ==================== RESET FUNCTIONS ====================

/**
 * Xóa kết quả phân tích đơn lẻ
 */
function clearSingleResults() {
    document.getElementById('resultText').textContent = 'Results will appear here...';
    document.getElementById('confidenceDisplay').innerHTML = '';
    document.getElementById('keywordsDisplay').innerHTML = '';
    document.getElementById('sms-input').value = '';
    document.getElementById('resultContainer').style.display = 'none';
}

/**
 * Xóa kết quả phân tích batch
 */
function clearBatchResults() {
    const batchResult = document.getElementById('batchResult');
    if (batchResult) {
        batchResult.innerHTML = '';
        batchResult.style.display = 'none';
    }
    
    // Reset file input
    const fileInput = document.getElementById('csv-file');
    if (fileInput) fileInput.value = '';
    
    // Reset file label hiển thị
    const fileLabel = document.querySelector('.file-label');
    const fileText = document.querySelector('.file-text');
    if (fileLabel && fileText) {
        fileText.textContent = 'Choose CSV file';
        fileLabel.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        fileLabel.style.background = 'rgba(255, 255, 255, 0.1)';
    }
}

// ==================== SINGLE PREDICTION HANDLER ====================

/**
 * Xử lý form phân tích tin nhắn đơn lẻ
 */
document.getElementById('single-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const smsInput = document.getElementById('sms-input').value.trim();
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
    const confidenceDisplay = document.getElementById('confidenceDisplay');
    const keywordsDisplay = document.getElementById('keywordsDisplay');
    
    // Validate input
    if (!smsInput) {
        showResult('Please enter a message to analyze!', 'warning');
        confidenceDisplay.innerHTML = '';
        keywordsDisplay.innerHTML = '';
        return;
    }
    
    try {
        // Hiển thị trạng thái loading
        showResult('🔄 Analyzing message...', 'loading');
        confidenceDisplay.innerHTML = '';
        keywordsDisplay.innerHTML = '';
        
        // Gọi API phân tích
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
        
        // Lấy độ tin cậy từ response API
        const confidence = data.confidence || 0;
        
        // Xác định class CSS cho độ tin cậy dựa trên % VÀ loại tin nhắn
        let confidenceClass = 'confidence-medium';
        if (data.label === 'spam') {
            // Tin nhắn spam - dùng màu đỏ/cam
            if (confidence >= 80) confidenceClass = 'confidence-high';
            else if (confidence >= 60) confidenceClass = 'confidence-medium';
            else confidenceClass = 'confidence-low';
        } else {
            // Tin nhắn hợp lệ - dùng màu xanh lá
            if (confidence >= 80) confidenceClass = 'confidence-high-ham';
            else if (confidence >= 60) confidenceClass = 'confidence-medium-ham';
            else confidenceClass = 'confidence-low-ham';
        }
        
        // Hiển thị kết quả dựa trên prediction
        if (data.label === 'spam') {
            showResult('🚨 SPAM DETECTED!', 'spam');
            
            // Hiển thị độ tin cậy với styling động
            confidenceDisplay.innerHTML = `
                <div class="result-confidence ${confidenceClass}">
                    Confidence: ${confidence}%
                </div>
            `;
            
            // LUÔN HIỂN THỊ TỪ KHÓA SPAM (nếu có)
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
            } else {
                keywordsDisplay.innerHTML = `
                    <div class="spam-keywords">
                        <h4>Spam pattern detected based on message content</h4>
                    </div>
                `;
            }
            
        } else {
            // Tin nhắn hợp lệ
            showResult('✅ SAFE MESSAGE', 'ham');
            
            // Hiển thị độ tin cậy với styling động
            confidenceDisplay.innerHTML = `
                <div class="result-confidence ${confidenceClass}">
                    Confidence: ${confidence}%
                </div>
            `;
            keywordsDisplay.innerHTML = ''; // Không hiển thị keywords cho ham
        }
        
    } catch (error) {
        console.error('Error:', error);
        showResult('❌ Connection error! Please check if backend is running.', 'error');
        confidenceDisplay.innerHTML = '';
        keywordsDisplay.innerHTML = '';
    }
});

// ==================== BATCH UPLOAD HANDLER - FIXED VERSION ====================

/**
 * Xử lý form upload và phân tích batch file CSV
 */
document.getElementById('batch-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('csv-file');
    const batchResult = document.getElementById('batchResult');
    
    // Validate file input
    if (!fileInput.files[0]) {
        batchResult.innerHTML = '<div class="result-warning">Please select a CSV file!</div>';
        batchResult.style.display = 'block';
        return;
    }
    
    try {
        // Cập nhật UI trạng thái processing
        const submitBtn = document.querySelector('#batch-form .submit-btn');
        submitBtn.textContent = '🔄 Processing...';
        submitBtn.disabled = true;
        
        batchResult.innerHTML = '<div class="loading">🔄 Processing CSV file...</div>';
        batchResult.style.display = 'block';
        
        // Chuẩn bị FormData để upload file
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        console.log('📤 Sending batch request to:', `${API_BASE_URL}/batch-predict-json`);
        console.log('📁 File:', fileInput.files[0].name);
        
        // GỌI API THẬT cho batch processing
        const response = await fetch(`${API_BASE_URL}/batch-predict-json`, {
            method: 'POST',
            body: formData
        });
        
        console.log('📥 Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('❌ Server error:', errorText);
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }
        
        const data = await response.json();
        console.log('✅ API Response data:', data);
        
        // Xử lý kết quả thành công
        if (data.success) {
            displayBatchResults(data);
        } else {
            throw new Error(data.error || 'Processing failed');
        }
        
    } catch (error) {
        console.error('❌ Batch upload error:', error);
        batchResult.innerHTML = `
            <div class="result-error">
                ❌ Batch processing error! ${error.message}
                <br><small>Check console for details</small>
            </div>
        `;
    } finally {
        // Khôi phục UI về trạng thái ban đầu
        const submitBtn = document.querySelector('#batch-form .submit-btn');
        submitBtn.textContent = 'Process Batch';
        submitBtn.disabled = false;
    }
});

// ==================== BATCH RESULTS DISPLAY - REAL VERSION ====================

/**
 * Hiển thị kết quả phân tích batch từ API thực tế
 * @param {Object} data - Dữ liệu kết quả từ API batch
 */
function displayBatchResults(data) {
    const batchResult = document.getElementById('batchResult');
    
    console.log("Displaying REAL batch results:", data);
    
    // Xây dựng HTML structure cho kết quả
    let resultsHTML = `
        <div class="batch-results-container">
            <div class="batch-results-header">
                <h3 class="batch-results-title">Batch Analysis Results</h3>
                <p class="batch-results-subtitle">Processing file: ${data.filename}</p>
            </div>
            
            <div class="batch-stats">
                <div class="stat-card">
                    <div class="stat-value total-stat">${data.total_messages}</div>
                    <div class="stat-label">Total Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value spam-stat">${data.spam_count}</div>
                    <div class="stat-label">Spam Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value ham-stat">${data.ham_count}</div>
                    <div class="stat-label">Legitimate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: #ffd700;">${data.spam_rate || 0}%</div>
                    <div class="stat-label">Spam Rate</div>
                </div>
            </div>
            
            <div class="tables-container">
    `;
    
    // Phân loại kết quả thành spam và ham
    const spamResults = data.results.filter(item => item.is_spam);
    const hamResults = data.results.filter(item => !item.is_spam);
    
    // ==================== SPAM TABLE ====================
    resultsHTML += `
        <div class="table-section spam-section">
            <div class="section-header">
                <h4 class="section-title">Spam Messages</h4>
                <span class="section-count">${spamResults.length} messages</span>
            </div>
    `;
    
    if (spamResults.length > 0) {
        resultsHTML += `
            <table class="batch-results-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Message Content</th>
                        <th>Spam Keywords</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Duyệt qua từng kết quả spam
        spamResults.forEach(item => {
            const displayMessage = item.text.length > 60 
                ? item.text.substring(0, 60) + '...' 
                : item.text;
            
            // Hiển thị từ spam nếu có
            let keywordsHTML = 'None detected';
            if (item.top_spam_words && item.top_spam_words.length > 0) {
                keywordsHTML = item.top_spam_words.map(word => 
                    `<span class="keyword-tag-small">${word[0]}</span>`
                ).join(' ');
            }
            
            resultsHTML += `
                <tr>
                    <td class="id-cell">${item.id || 'N/A'}</td>
                    <td>${displayMessage}</td>
                    <td class="keywords-cell">${keywordsHTML}</td>
                    <td class="confidence-cell">${item.confidence || 'N/A'}%</td>
                </tr>
            `;
        });
        
        resultsHTML += `
                </tbody>
            </table>
        `;
    } else {
        resultsHTML += `
            <div style="text-align: center; padding: 40px; color: rgba(255, 255, 255, 0.6); font-style: italic;">
                No spam messages detected in this file
            </div>
        `;
    }
    
    resultsHTML += `</div>`; // Đóng spam-section
    
    // ==================== HAM TABLE ====================
    resultsHTML += `
        <div class="table-section ham-section">
            <div class="section-header">
                <h4 class="section-title">Legitimate Messages</h4>
                <span class="section-count">${hamResults.length} messages</span>
            </div>
    `;
    
    if (hamResults.length > 0) {
        resultsHTML += `
            <table class="batch-results-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Message Content</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Duyệt qua từng kết quả ham
        hamResults.forEach(item => {
            const displayMessage = item.text.length > 60 
                ? item.text.substring(0, 60) + '...' 
                : item.text;
            
            resultsHTML += `
                <tr>
                    <td class="id-cell">${item.id || 'N/A'}</td>
                    <td>${displayMessage}</td>
                    <td class="confidence-cell">${item.confidence || 'N/A'}%</td>
                </tr>
            `;
        });
        
        resultsHTML += `
                </tbody>
            </table>
        `;
    } else {
        resultsHTML += `
            <div style="text-align: center; padding: 40px; color: rgba(255, 255, 255, 0.6); font-style: italic;">
                No legitimate messages found in this file
            </div>
        `;
    }
    
    resultsHTML += `</div>`; // Đóng ham-section
    
    // Đóng containers
    resultsHTML += `
            </div>
        </div>
    `;
    
    batchResult.innerHTML = resultsHTML;
}

// ==================== FILE INPUT HANDLER ====================

/**
 * Xử lý sự kiện thay đổi file input để cập nhật UI
 */
document.getElementById('csv-file').addEventListener('change', function(e) {
    const fileLabel = document.querySelector('.file-label');
    const fileText = document.querySelector('.file-text');
    
    if (this.files[0]) {
        // Cập nhật UI khi có file được chọn
        fileText.textContent = this.files[0].name;
        fileLabel.style.borderColor = '#4ade80';
        fileLabel.style.background = 'rgba(74, 222, 128, 0.1)';
    } else {
        // Reset UI khi không có file
        fileText.textContent = 'Choose CSV file';
        fileLabel.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        fileLabel.style.background = 'rgba(255, 255, 255, 0.1)';
    }
});

// ==================== RESULT DISPLAY FUNCTION ====================

/**
 * Hiển thị kết quả với styling phù hợp
 * @param {string} message - Thông báo cần hiển thị
 * @param {string} type - Loại kết quả ('spam', 'ham', 'error', 'warning', 'loading')
 */
function showResult(message, type) {
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
    
    // Reset classes
    resultContainer.className = 'result-container';
    resultText.className = '';
    
    // Áp dụng styling theo loại kết quả
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

// ==================== APPLICATION INITIALIZATION ====================

/**
 * Khởi tạo ứng dụng khi DOM đã load xong
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 AmongSMS Frontend Initialized');
    testBackendConnection();
    
    // Thêm CSS bổ sung cho batch results
    const additionalCSS = `
        .keyword-tag-small {
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            padding: 2px 6px;
            border-radius: 8px;
            font-size: 0.7rem;
            border: 1px solid rgba(239, 68, 68, 0.3);
            margin: 1px;
            display: inline-block;
        }
        .keywords-cell {
            max-width: 150px;
            min-width: 120px;
        }
        .confidence-cell {
            font-weight: bold;
            color: #ffd700;
            text-align: center;
            width: 80px;
        }
        .id-cell {
            font-weight: bold;
            color: #ffd700;
            text-align: center;
            width: 60px;
        }
        /* CSS classes cho độ tin cậy của tin nhắn hợp lệ (màu xanh) */
        .confidence-high-ham {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .confidence-medium-ham {
            background: rgba(34, 197, 94, 0.15);
            color: #16a34a;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        .confidence-low-ham {
            background: rgba(34, 197, 94, 0.1);
            color: #15803d;
            border: 1px solid rgba(34, 197, 94, 0.15);
        }
    `;
    
    const styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = additionalCSS;
    document.head.appendChild(styleSheet);
});

// ==================== BACKEND CONNECTION TEST ====================

/**
 * Kiểm tra kết nối đến backend server
 */
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