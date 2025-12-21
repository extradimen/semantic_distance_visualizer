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

    // 拖拽上传 - 阻止浏览器默认的拖拽行为（下载文件）
    // 首先在整个文档级别阻止默认拖拽行为
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
    }, false);
    
    document.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
    }, false);
    
    const fileLabelElement = document.querySelector('.file-label');
    
    fileLabelElement.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.style.background = '#f0f4ff';
        return false;
    }, false);

    fileLabelElement.addEventListener('dragenter', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.style.background = '#f0f4ff';
        return false;
    }, false);

    fileLabelElement.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.style.background = '#ffffff';
        return false;
    }, false);

    fileLabelElement.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        this.style.background = '#ffffff';
        
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            // 使用 DataTransfer 来设置文件
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(files[0]);
            fileInput.files = dataTransfer.files;
            fileLabel.textContent = files[0].name;
        }
        return false;
    }, false);

    // 提交按钮点击事件（改为按钮点击而不是表单提交）
    submitBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // 检查文件是否已选择
        if (!fileInput.files || fileInput.files.length === 0) {
            showError('请先选择要上传的文件');
            return;
        }
        
        // 检查阈值是否有效
        const thresholdInput = document.getElementById('threshold');
        const threshold = parseFloat(thresholdInput.value);
        if (isNaN(threshold) || threshold < 0 || threshold > 1) {
            showError('阈值必须是0到1之间的数字');
            return;
        }
        
        // 检查幂次是否有效
        const powerInput = document.getElementById('power');
        const power = parseFloat(powerInput.value);
        if (isNaN(power) || power < 1 || power > 10) {
            showError('幂次必须是1到10之间的数字');
            return;
        }
        
        const formData = new FormData(form);
        formData.append('threshold', threshold);
        formData.append('power', power);

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

            // 先检查响应状态
            if (!response.ok) {
                // 尝试解析错误信息
                let errorMessage = '上传失败';
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                } catch (e) {
                    // 如果无法解析JSON，使用状态文本
                    errorMessage = `服务器错误: ${response.status} ${response.statusText}`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();

            // 显示结果
            displayResults(data);
            
        } catch (err) {
            console.error('上传错误:', err);
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

        // 显示热力图（可能是多个）
        const heatmapContainer = document.getElementById('heatmapContainer');
        heatmapContainer.innerHTML = '';
        
        // 如果热力图是数组（多个图），否则是单个
        if (Array.isArray(data.heatmap)) {
            data.heatmap.forEach(url => {
                const img = document.createElement('img');
                img.src = url;
                img.alt = '热力图';
                img.style.width = '100%';
                img.style.height = 'auto';
                img.style.marginBottom = '20px';
                img.style.borderRadius = '6px';
                img.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.05)';
                heatmapContainer.appendChild(img);
            });
        } else {
            const img = document.createElement('img');
            img.src = data.heatmap;
            img.alt = '热力图';
            img.style.width = '100%';
            img.style.height = 'auto';
            img.style.borderRadius = '6px';
            img.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.05)';
            heatmapContainer.appendChild(img);
        }

        // 显示MDS图
        if (document.getElementById('mdsImg')) {
            document.getElementById('mdsImg').src = data.mds;
        }

        // 显示3D MDS图
        const mds3dIframe = document.getElementById('mds3dIframe');
        if (mds3dIframe && data.mds_3d) {
            mds3dIframe.src = data.mds_3d;
            mds3dIframe.onload = function() {
                console.log('3D MDS图加载完成');
            };
        }

        // 显示网络图（两个静态图）
        const networkDegreeImg = document.getElementById('networkDegreeImg');
        if (networkDegreeImg && data.network_degree) {
            networkDegreeImg.src = data.network_degree;
            networkDegreeImg.onload = function() {
                console.log('加权度中心性网络图加载完成');
            };
        }
        
        const networkEigenImg = document.getElementById('networkEigenImg');
        if (networkEigenImg && data.network_eigen) {
            networkEigenImg.src = data.network_eigen;
            networkEigenImg.onload = function() {
                console.log('加权特征向量中心性网络图加载完成');
            };
        }

        // 显示相似度表格
        const tableBody = document.getElementById('similarityTableBody');
        tableBody.innerHTML = '';
        
        if (data.similarity_data && data.similarity_data.top_pairs) {
            data.similarity_data.top_pairs.forEach((pair, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${index + 1}</td>
                    <td>${pair.word1}</td>
                    <td>${pair.category1 || '-'}</td>
                    <td>${pair.word2}</td>
                    <td>${pair.category2 || '-'}</td>
                    <td><strong>${pair.similarity.toFixed(4)}</strong></td>
                `;
                tableBody.appendChild(row);
            });
        }

        // 显示Excel下载按钮
        const downloadExcelBtn = document.getElementById('downloadExcelBtn');
        if (downloadExcelBtn && data.excel_download) {
            downloadExcelBtn.style.display = 'inline-block';
            downloadExcelBtn.onclick = function() {
                window.location.href = data.excel_download;
            };
        }

        // 显示结果区域
        results.classList.remove('hidden');
        
        // 滚动到结果区域（使用 requestAnimationFrame 优化性能）
        requestAnimationFrame(() => {
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    }


    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        error.classList.remove('hidden');
        error.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
