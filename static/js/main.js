document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.querySelector('.file-text');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const submitBtn = document.getElementById('submitBtn');

    // 文件选择处理
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            fileLabel.textContent = e.target.files[0].name;
        }
    });

    // 拖拽上传
    const fileLabelElement = document.querySelector('.file-label');
    fileLabelElement.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.background = '#e8e8ff';
    });

    fileLabelElement.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.background = 'white';
    });

    fileLabelElement.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.background = 'white';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            fileLabel.textContent = files[0].name;
        }
    });

    // 表单提交
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const threshold = document.getElementById('threshold').value;
        formData.append('threshold', threshold);

        // 隐藏之前的结果和错误
        results.classList.add('hidden');
        error.classList.add('hidden');
        
        // 显示加载状态
        loading.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.querySelector('.btn-text').textContent = '⏳ 分析中...';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || '上传失败');
            }

            // 显示结果
            displayResults(data);
            
        } catch (err) {
            showError(err.message);
        } finally {
            loading.classList.add('hidden');
            submitBtn.disabled = false;
            submitBtn.querySelector('.btn-text').textContent = '🚀 开始分析';
        }
    });

    function displayResults(data) {
        // 更新词汇数量
        document.getElementById('wordCount').textContent = data.word_count;

        // 显示图片
        document.getElementById('heatmapImg').src = data.heatmap;
        document.getElementById('networkImg').src = data.network;
        document.getElementById('mdsImg').src = data.mds;

        // 显示相似度表格
        const tableBody = document.getElementById('similarityTableBody');
        tableBody.innerHTML = '';
        
        if (data.similarity_data && data.similarity_data.top_pairs) {
            data.similarity_data.top_pairs.forEach((pair, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${pair.word1}</td>
                    <td>${pair.word2}</td>
                    <td><strong>${pair.similarity.toFixed(4)}</strong></td>
                `;
                tableBody.appendChild(row);
            });
        }

        // 显示结果区域
        results.classList.remove('hidden');
        
        // 滚动到结果区域
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        error.classList.remove('hidden');
        error.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});

