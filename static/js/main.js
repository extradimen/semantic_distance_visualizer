document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.querySelector('.file-text');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const submitBtn = document.getElementById('submitBtn');
    
    let networkData = null; // 存储网络数据用于动态更新
    let network = null; // vis-network实例

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
        this.style.background = '#f0f4ff';
    });

    fileLabelElement.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.background = '#ffffff';
    });

    fileLabelElement.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.style.background = '#ffffff';
        
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

        // 存储网络数据用于动态更新
        networkData = data.network_data;

        // 检查网络数据是否存在
        if (!data.network_data) {
            console.warn('警告: 没有收到网络数据，交互式网络图可能无法使用');
            const networkContainer = document.getElementById('networkContainer');
            if (networkContainer) {
                networkContainer.innerHTML = '<div style="padding: 20px; text-align: center; color: #64748b;">无法加载交互式网络图：服务器未返回网络数据</div>';
            }
        } else {
            // 设置交互式网络图
            const threshold = parseFloat(document.getElementById('threshold').value) || 0.3;
            setupDynamicNetwork(data.network_data, threshold);
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

        // 显示结果区域
        results.classList.remove('hidden');
        
        // 滚动到结果区域
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function setupDynamicNetwork(encodedData, initialThreshold) {
        if (!encodedData) {
            console.error('没有网络数据，无法设置交互式网络图');
            const container = document.getElementById('networkContainer');
            if (container) {
                container.innerHTML = '<div style="padding: 20px; text-align: center; color: #64748b;">无法加载交互式网络图：缺少网络数据</div>';
            }
            return;
        }

        const container = document.getElementById('networkContainer');
        const thresholdSlider = document.getElementById('networkThreshold');
        const thresholdValue = document.getElementById('thresholdValue');
        const loadingIndicator = document.getElementById('networkLoading');
        const legendContainer = document.getElementById('networkLegend');

        // 设置初始阈值
        thresholdSlider.value = initialThreshold;
        thresholdValue.textContent = parseFloat(initialThreshold).toFixed(2);

        // 防抖函数
        let debounceTimer = null;
        const debounceDelay = 300;

        // 更新网络图的函数
        const updateNetwork = async (threshold) => {
            if (debounceTimer) {
                clearTimeout(debounceTimer);
            }

            debounceTimer = setTimeout(async () => {
                loadingIndicator.classList.remove('hidden');
                
                try {
                    if (!encodedData) {
                        throw new Error('网络数据未初始化');
                    }

                    const response = await fetch('/network_data', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            threshold: threshold,
                            network_data: encodedData
                        })
                    });

                    if (!response.ok) {
                        const errorText = await response.text();
                        let errorMsg = '获取网络数据失败';
                        try {
                            const errorData = JSON.parse(errorText);
                            errorMsg = errorData.error || errorMsg;
                        } catch (e) {
                            errorMsg = `HTTP ${response.status}: ${errorText.substring(0, 100)}`;
                        }
                        throw new Error(errorMsg);
                    }

                    const data = await response.json();

                    if (data.error) {
                        throw new Error(data.error);
                    }

                    // 更新可视化
                    updateNetworkVisualization(data, container, legendContainer);
                    
                } catch (err) {
                    console.error('更新网络图失败:', err);
                    showError('更新网络图失败: ' + err.message);
                } finally {
                    loadingIndicator.classList.add('hidden');
                }
            }, debounceDelay);
        };

        // 监听滑块变化
        thresholdSlider.addEventListener('input', function(e) {
            const threshold = parseFloat(e.target.value);
            thresholdValue.textContent = threshold.toFixed(2);
            updateNetwork(threshold);
        });

        // 初始加载
        updateNetwork(initialThreshold);
    }

    function updateNetworkVisualization(data, container, legendContainer) {
        // 创建节点和边的数据集
        const nodes = new vis.DataSet(data.nodes);
        const edges = new vis.DataSet(data.edges);

        // 配置选项 - VOSviewer风格
        const options = {
            nodes: {
                shape: 'dot',
                size: 20,
                font: {
                    size: 14,
                    color: '#2c3e50',
                    face: 'Arial, sans-serif',
                    bold: {
                        mod: 'bold'
                    }
                },
                borderWidth: 2,
                borderWidthSelected: 3,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.2)',
                    size: 5,
                    x: 2,
                    y: 2
                },
                chosen: {
                    node: function(values, id, selected, hovering) {
                        if (hovering) {
                            values.size = values.size * 1.2;
                        }
                    }
                }
            },
            edges: {
                width: 2,
                color: {
                    color: '#848484',
                    highlight: '#3b82f6',
                    hover: '#3b82f6'
                },
                smooth: {
                    type: 'continuous',
                    roundness: 0.5
                },
                arrows: {
                    to: {
                        enabled: false
                    }
                },
                selectionWidth: 3,
                shadow: {
                    enabled: true,
                    color: 'rgba(0,0,0,0.1)',
                    size: 3
                }
            },
            physics: {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 200,
                    fit: true
                },
                barnesHut: {
                    gravitationalConstant: -2000,
                    centralGravity: 0.1,
                    springLength: 200,
                    springConstant: 0.04,
                    damping: 0.09,
                    avoidOverlap: 0.5
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true,
                selectConnectedEdges: true
            },
            layout: {
                improvedLayout: true,
                hierarchical: {
                    enabled: false
                }
            }
        };

        // 创建网络
        const networkData = {
            nodes: nodes,
            edges: edges
        };

        if (network) {
            network.destroy();
        }

        network = new vis.Network(container, networkData, options);

        // 生成图例
        if (data.category_colors && legendContainer) {
            legendContainer.innerHTML = '';
            Object.keys(data.category_colors).forEach(category => {
                const legendItem = document.createElement('div');
                legendItem.className = 'legend-item';
                legendItem.innerHTML = `
                    <div class="legend-color" style="background-color: ${data.category_colors[category]}"></div>
                    <span class="legend-label">${category}</span>
                `;
                legendContainer.appendChild(legendItem);
            });
        }
    }

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        error.classList.remove('hidden');
        error.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});
