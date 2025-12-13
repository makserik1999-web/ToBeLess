// static/js/script.js
// Extended for: heatmap, escalation warnings, conflict types, hazards, hotspots
class FDApp {
    constructor(opts={}) {
        console.log('[FDApp] Initializing application...');
        this.pollIntervalMs = opts.pollIntervalMs || 800;
        this.isStreaming = false;
        this.isAnalyzing = false;
        this.analysisJobId = null;
        this.pollTimer = null;
        this.uiAlertsSeen = new Set();

        this.el = id => document.getElementById(id);
        this.elems = {
            sourceType: this.el('sourceType'),
            sourceInput: this.el('sourceInput'),
            fileInput: this.el('fileInput'),
            startBtn: this.el('startBtn'),
            stopBtn: this.el('stopBtn'),
            statusAlert: this.el('statusAlert'),
            videoPlaceholder: this.el('videoPlaceholder'),
            videoFeed: this.el('videoFeed'),
            uploadedVideo: this.el('uploadedVideo'),
            uploadedVideoSource: this.el('uploadedVideoSource'),
            peopleCount: this.el('peopleCount'),
            fightCount: this.el('fightCount'),
            confidence: this.el('confidence'),
            fps: this.el('fps'),
            eventLog: this.el('eventLog'),
            segmentsGallery: this.el('segmentsGallery'),
            updateSettingsBtn: this.el('updateSettings'),
            bodyThreshold: this.el('bodyThreshold'),
            limbThreshold: this.el('limbThreshold'),
            holdDuration: this.el('holdDuration'),
            statusIndicator: this.el('statusIndicator'),
            peopleChart: this.el('peopleChart'),
            confidenceChart: this.el('confidenceChart'),
            // NEW
            escalationBanner: this.el('escalationBanner'),
            conflictTypeBadge: this.el('conflictTypeBadge'),
            hazardsList: this.el('hazardsList'),
            heatmapImg: this.el('heatmapImg'),
            hotspotsList: this.el('hotspotsList')
        };

        this.setupUI();
        this.startPolling();
        this.connectSSE();
        this._initCharts();

        // periodic heatmap refresh (only when streaming/has events)
        this.heatmapTimer = setInterval(()=> this.refreshHeatmap(), 2500);

        console.log('[FDApp] Initialization complete');
        console.log('[FDApp] Start button element:', this.elems.startBtn);
    }

    connectSSE() {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.eventSource = new EventSource('/stats_stream');
        
        this.eventSource.onmessage = (event) => {
            try {
                const stats = JSON.parse(event.data);

                // Обновляем статистику
                if (this.elems.peopleCount) 
                    this.elems.peopleCount.textContent = stats.people || 0;
                if (this.elems.fightCount) 
                    this.elems.fightCount.textContent = stats.fights || 0;
                if (this.elems.confidence) 
                    this.elems.confidence.textContent = `${Math.round(stats.confidence || 0)}%`;
                if (this.elems.fps) 
                    this.elems.fps.textContent = stats.fps || 0;

                // Обновляем графики в реальном времени
                this.appendChart(stats.people || 0, Math.round(stats.confidence || 0));

                // Индикатор статуса
                if (this.elems.statusIndicator) {
                    if (stats.fights > 0) {
                        this.elems.statusIndicator.className = 'status-indicator status-fight';
                        this.elems.statusIndicator.textContent = ' Fight detected';
                    } else if (stats.escalation_warning) {
                        this.elems.statusIndicator.className = 'status-indicator status-warning';
                        this.elems.statusIndicator.textContent = ' Escalation warning';
                    } else {
                        this.elems.statusIndicator.className = 'status-indicator status-normal';
                        this.elems.statusIndicator.textContent = ' System Ready';
                    }
                }
            } catch(e) {
                console.warn('SSE parse error:', e);
            }
        };
        
        this.eventSource.onerror = (err) => {
            console.warn('SSE connection error, reconnecting...', err);
            setTimeout(() => this.connectSSE(), 3000);
        };
        }

    setupUI(){
        if (this.elems.sourceType) {
            this.elems.sourceType.addEventListener('change', () => {
                const v = this.elems.sourceType.value;
                if (v === 'video') {
                    if (this.elems.fileInput) this.elems.fileInput.style.display = 'inline-block';
                    if (this.elems.sourceInput) this.elems.sourceInput.style.display = 'none';
                } else if (v === 'rtsp') {
                    if (this.elems.fileInput) this.elems.fileInput.style.display = 'none';
                    if (this.elems.sourceInput) { this.elems.sourceInput.style.display = 'inline-block'; this.elems.sourceInput.placeholder='rtsp/http URL'; }
                } else {
                    if (this.elems.fileInput) this.elems.fileInput.style.display = 'none';
                    if (this.elems.sourceInput) this.elems.sourceInput.style.display = 'none';
                }
            });
        }

        if (this.elems.startBtn) this.elems.startBtn.addEventListener('click', ()=>this.start());
        if (this.elems.stopBtn) this.elems.stopBtn.addEventListener('click', ()=>this.stop());
        if (this.elems.updateSettingsBtn) this.elems.updateSettingsBtn.addEventListener('click', ()=>this.updateSettings());
        if (this.el('clearEvents')) this.el('clearEvents').addEventListener('click', ()=>{ if(this.elems.eventLog) this.elems.eventLog.innerHTML = '<div class="event-placeholder"><p>События пока не зафиксированы</p><small>Система будет автоматически регистрировать все обнаруженные инциденты</small></div>'; });

        // Face recognition and blur buttons
        const faceRecBtn = this.el('faceRecognitionBtn');
        const faceBlurBtn = this.el('faceBlurBtn');

        if (faceRecBtn) {
            faceRecBtn.addEventListener('click', async () => {
                await this.toggleFaceRecognition(faceRecBtn);
            });
        }

        if (faceBlurBtn) {
            faceBlurBtn.addEventListener('click', async () => {
                await this.toggleFaceBlur(faceBlurBtn);
            });
        }

        // Load initial feature status on page load
        this.loadFeatureStatus();

        // file change -> show local preview
        if (this.elems.fileInput) {
            this.elems.fileInput.addEventListener('change', (e)=>{
                const f = e.target.files && e.target.files[0];
                if (f) this.showVideoPreview(f);
            });
        }
    }

    _initCharts(){
        try {
            const pCtx = this.elems.peopleChart.getContext('2d');
            const cCtx = this.elems.confidenceChart.getContext('2d');
            this.peopleData = {labels:[], datasets:[{label:'People', data:[], fill:false, tension:0.3}]};
            this.confData = {labels:[], datasets:[{label:'Confidence', data:[], fill:true, tension:0.3}]};
            this.peopleChart = new Chart(pCtx, {type:'line', data:this.peopleData, options:{animation:false, responsive:true, plugins:{legend:{display:false}}}});
            this.confChart = new Chart(cCtx, {type:'line', data:this.confData, options:{animation:false, responsive:true, plugins:{legend:{display:false}}}});
        } catch(e){
            console.warn('Chart init failed', e);
        }
    }

    appendChart(pointPeople, pointConf){
        const t = new Date().toLocaleTimeString();
        if (this.peopleChart) {
            this.peopleData.labels.push(t);
            this.peopleData.datasets[0].data.push(pointPeople);
            if (this.peopleData.labels.length>60) { this.peopleData.labels.shift(); this.peopleData.datasets[0].data.shift(); }
            this.peopleChart.update('none');
        }
        if (this.confChart) {
            this.confData.labels.push(t);
            this.confData.datasets[0].data.push(pointConf);
            if (this.confData.labels.length>60) { this.confData.labels.shift(); this.confData.datasets[0].data.shift(); }
            this.confChart.update('none');
        }
    }

    showVideoPreview(file) {
        try {
            const url = URL.createObjectURL(file);
            this.hideAllVideoElements();
            if (this.elems.uploadedVideoSource) {
                this.elems.uploadedVideoSource.src = url;
            }
            if (this.elems.uploadedVideo) {
                this.elems.uploadedVideo.load();
                this.elems.uploadedVideo.style.display = 'block';
            }
            this.localPreviewUrl = url;
        } catch(e) {
            console.error('Failed to create video preview:', e);
        }
    }

    hideAllVideoElements() {
        if (this.elems.videoPlaceholder) this.elems.videoPlaceholder.style.display = 'none';
        if (this.elems.videoFeed) this.elems.videoFeed.style.display = 'none';
        if (this.elems.uploadedVideo) this.elems.uploadedVideo.style.display = 'none';
    }

    showVideoPlaceholder() {
        this.hideAllVideoElements();
        if (this.elems.videoPlaceholder) this.elems.videoPlaceholder.style.display = 'block';
    }

    async start(){
        console.log('[FDApp] Start button clicked');

        if (this.elems.startBtn) {
            this.elems.startBtn.disabled = true;
            this.elems.startBtn.textContent = 'Starting…';
        }

        // Clear charts
        if (this.peopleChart) {
            this.peopleData.labels = [];
            this.peopleData.datasets[0].data = [];
            this.peopleChart.update();
        }
        if (this.confChart) {
            this.confData.labels = [];
            this.confData.datasets[0].data = [];
            this.confChart.update();
        }

        try {
            const st = this.elems.sourceType ? this.elems.sourceType.value : '0';
            console.log('[FDApp] Source type:', st);

            if (st === 'video') {
                if (!this.elems.fileInput || !this.elems.fileInput.files || this.elems.fileInput.files.length===0) {
                    throw new Error('No file selected');
                }

                const fd = new FormData();
                fd.append('file', this.elems.fileInput.files[0]);
                console.log('[FDApp] Uploading file:', this.elems.fileInput.files[0].name);

                const resp = await fetch('/start_stream', {method:'POST', body: fd});
                const j = await this._parseJSON(resp);
                console.log('[FDApp] Upload response:', j);

                if (!j || !j.success) throw new Error(j && j.error ? j.error : 'Upload failed');

                this.isAnalyzing = true;
                this.analysisJobId = j.job_id || null;

                this.hideAllVideoElements();
                if (this.elems.videoFeed) {
                    this.elems.videoFeed.src = '/video_feed?t=' + Date.now();
                    this.elems.videoFeed.style.display = 'block';
                }
                this.isStreaming = true;
                if (this.elems.stopBtn) { this.elems.stopBtn.disabled = false; }
                this.showAlert('File uploaded — showing processed stream', 'success', 3500);
            } else {
                const source = (st === '0') ? '0' : (this.elems.sourceInput && this.elems.sourceInput.value) || '';
                console.log('[FDApp] Starting stream with source:', source);

                const resp = await fetch('/start_stream', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({source})
                });
                const j = await this._parseJSON(resp);
                console.log('[FDApp] Stream start response:', j);

                if (!j || !j.success) throw new Error(j && j.error ? j.error : 'Start stream failed');

                this.hideAllVideoElements();
                if (this.elems.videoFeed) {
                    this.elems.videoFeed.src = '/video_feed?t=' + Date.now();
                    this.elems.videoFeed.style.display = 'block';
                }
                this.isStreaming = true;
                if (this.elems.stopBtn) { this.elems.stopBtn.disabled = false; }
                this.showAlert('Stream started', 'success', 3000);
                console.log('[FDApp] Stream started successfully');
            }
        } catch (err) {
            console.error('[FDApp] Start failed:', err);
            this.showAlert('Start failed: '+(err.message||err),'error',8000);
            if (this.elems.stopBtn) this.elems.stopBtn.disabled = true;
        } finally {
            if (this.elems.startBtn) {
                this.elems.startBtn.disabled = false;
                this.elems.startBtn.textContent = 'Начать детекцию';
            }
        }
    }

    async stop(){
        try {
            if (this.elems.stopBtn) { this.elems.stopBtn.disabled = true; }
            const resp = await fetch('/stop_stream', {method:'POST'});
            
            const j = await this._parseJSON(resp);
            if (!j || !j.success) throw new Error(j && j.error ? j.error : 'Stop failed');

            this.hideAllVideoElements();
            this.showVideoPlaceholder();
            if (this.localPreviewUrl) { try{ URL.revokeObjectURL(this.localPreviewUrl); }catch(e){} this.localPreviewUrl = null; }

            this.isStreaming = false;
            this.isAnalyzing = false;
            this.analysisJobId = null;
            if (this.elems.stopBtn) this.elems.stopBtn.disabled = true;
            this.showAlert('Stopped', 'success', 3000);
        } catch (err) {
            console.error(err);
            this.showAlert('Stop failed: '+(err.message||err),'error',8000);
            if (this.elems.stopBtn) this.elems.stopBtn.disabled = false;
        }
    }

    async pollJobOnce(jobId){
        try {
            const r = await fetch(`/job/${encodeURIComponent(jobId)}`);
            const j = await this._parseJSON(r);
            if (!j) return;
            if (j.analysis) {
                this.isAnalyzing = false; this.analysisJobId = null;
                this.handleAnalysisResult(j.analysis);
                this.showAlert('Background analysis finished', 'success', 4000);
            }
        } catch (e) { console.warn('pollJobOnce failed', e); }
    }

    startPolling(){
        if (this.pollTimer) return;
        this.pollTimer = setInterval(()=>this.poll(), this.pollIntervalMs);
    }
    stopPolling(){
        if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    }

    async poll(){
        if (!this.isStreaming) {
            return; // Не обновляем если стрим не активен
        }
        try {
            if (this.isAnalyzing && this.analysisJobId) {
                await this.pollJobOnce(this.analysisJobId);
                return;
            }
            const r = await fetch('/analytics');
            const j = await this._parseJSON(r);
            if (!j || !j.success) return;

            const recent = Array.isArray(j.recent_data) ? j.recent_data : (j.recent_data || []);
            const latest = (recent && recent.length) ? recent[recent.length - 1] : null;

            // derive people
            let people = 0;
            if (latest && (latest.people !== undefined)) {
                people = latest.people;
            } else if (j.analytics && Array.isArray(j.analytics.people_count_history) && j.analytics.people_count_history.length) {
                const lastEntry = j.analytics.people_count_history[j.analytics.people_count_history.length - 1];
                people = lastEntry && lastEntry.count !== undefined ? lastEntry.count : 0;
            } else if (j.latest_stats && j.latest_stats.people !== undefined) {
                people = j.latest_stats.people;
            }

            // fight flag & confidence
            let fight = false;
            if (latest && latest.fight !== undefined) fight = !!latest.fight;
            else if (j.latest_stats && j.latest_stats.fights !== undefined) fight = !!j.latest_stats.fights;

            let conf = 0;
            if (latest && latest.metrics && latest.metrics.confidence !== undefined) conf = latest.metrics.confidence;
            else if (j.latest_stats && j.latest_stats.confidence !== undefined) conf = j.latest_stats.confidence;

            if (this.elems.peopleCount) this.elems.peopleCount.textContent = stats.people || 0;
            if (this.elems.fightCount) this.elems.fightCount.textContent = stats.fights || 0;
            if (this.elems.confidence) this.elems.confidence.textContent = `${Math.round(stats.confidence || 0)}%`;
            if (this.elems.fps) this.elems.fps.textContent = stats.fps || 0;

            // status indicator
            if (this.elems.statusIndicator) {
                   // update hazards UI
                this._updateHazardsUI(j.latest_stats ? j.latest_stats.hazards : (j.analytics ? j.analytics.hazards : {}));
                    // update heatmap + hotspots every ~2s (cheap)
                if (!this.heatmapTimer || Date.now() - (this._lastHeatmap || 0) > 2000) {
                    this._lastHeatmap = Date.now();
                    this._updateHeatmapAndHotspots();
                }
            // escalation banner
            const escalation = latest && latest.metrics && latest.metrics.escalation_warning ? latest.metrics.escalation_warning : (j.latest_stats ? (j.latest_stats.escalation_warning ? {active:true, reason:'server'} : {active:false}) : {active:false});
            this._updateEscalationBanner(escalation);
                
            }

            // chart
            this.appendChart(people, Math.round(conf));

            // Show escalation (from latest_stats or latest.metrics)
            const escalationActive = j.latest_stats && j.latest_stats.escalation_warning ? j.latest_stats.escalation_warning : (latest && latest.escalation_warning && latest.escalation_warning.active);
            const tension = (latest && latest.tension_score) ? latest.tension_score : (j.latest_stats && j.latest_stats.confidence ? j.latest_stats.confidence : 0);
            this.renderEscalation(escalationActive, tension);

            // conflict type
            const ctype = (latest && latest.conflict_type) ? latest.conflict_type : (j.latest_stats && j.latest_stats.conflict_type ? j.latest_stats.conflict_type : '—');
            if (this.elems.conflictTypeBadge) this.elems.conflictTypeBadge.textContent = `Тип: ${ctype.replace('_',' ')}`;

            // hazards
            const hazards = (latest && latest.hazards) ? latest.hazards : (j.latest_stats && j.latest_stats.hazards ? j.latest_stats.hazards : (j.analytics && j.analytics.hazards ? j.analytics.hazards : {}));
            this.renderHazards(hazards);

            // if server finished streaming and returned analytics -> show final results once
            if (!j.streaming && j.analytics) {
                if (!this.isAnalyzing) {
                    try { this.handleAnalysisResult({ analytics: j.analytics }); } catch(e){ console.warn('final handle failed', e); }
                }
            }

            // UI alert for live fight
            if (latest && latest.fight) {
                const key = `live-${latest.frame}-${Math.round(Date.parse(latest.timestamp || new Date())/1000)}`;
                if (!this.uiAlertsSeen.has(key)) {
                    this.uiAlertsSeen.add(key);
                    this.prependEventLog({start_time: new Date().toLocaleString(), duration:0, confidence: latest.metrics ? latest.metrics.confidence : conf});
                    this.showAlert(`🚨 Live: possible fight — ${Math.round(conf)}%`,'error',5000);
                }
            }
        } catch (e) {
            // ignore network parse errors silently
        }
    }

    renderEscalation(active, tension){
        if (!this.elems.escalationBanner) return;
        if (active) {
            const level = tension > 60 ? 'escalation-high' : (tension > 35 ? 'escalation-mid' : 'escalation-low');
            this.elems.escalationBanner.className = level;
            this.elems.escalationBanner.style.display = 'inline-block';
            this.elems.escalationBanner.textContent = `⚠️ Escalation predicted • Tension ${Math.round(tension)}`;
        } else {
            this.elems.escalationBanner.style.display = 'none';
        }
    }

    renderHazards(hazards){
        if (!this.elems.hazardsList) return;
        try {
            const keys = Object.keys(hazards || {});
            if (!keys.length) { this.elems.hazardsList.innerHTML = '<div>Нет</div>'; return; }
            const parts = [];
            keys.forEach(k=>{
                const arr = hazards[k] || [];
                if (!arr.length) return;
                const cnt = arr.length;
                parts.push(`<div class="hazard-item"><div class="h-type">${k.toUpperCase()}</div><div class="h-count">${cnt} шт.</div></div>`);
            });
            this.elems.hazardsList.innerHTML = parts.join('');
            // fetch hotspots when hazards appear (or periodically)
            if (this.elems.hotspotsList) this.fetchHotspots();
        } catch(e){ console.warn('renderHazards', e); }
    }

    async refreshHeatmap(){
        if (!this.elems.heatmapImg) return;
        try {
            // cache-buster
            const res = await fetch('/heatmap');
            if (!res.ok) return;
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            this.elems.heatmapImg.src = url;
            // revoke previous after small delay
            setTimeout(()=>{ try{ URL.revokeObjectURL(url); }catch(e){} }, 8000);
        } catch(e){ /* ignore */ }
    }

    async fetchHotspots(){
        if (!this.elems.hotspotsList) return;
        try {
            const r = await fetch('/hotspots');
            const j = await this._parseJSON(r);
            if (!j || !j.success) { this.elems.hotspotsList.innerHTML = '<div>Нет данных</div>'; return; }
            const spots = j.hotspots || [];
            if (!spots.length) { this.elems.hotspotsList.innerHTML = '<div>Нет горячих точек</div>'; return; }
            this.elems.hotspotsList.innerHTML = spots.map(s=>`<div class="hotspot-item">x:${s.x}, y:${s.y} • intensity:${(s.intensity*100).toFixed(0)}% • events:${s.events}</div>`).join('');
        } catch(e){ console.warn('fetchHotspots', e); }
    }

    handleAnalysisResult(analysis){
        try {
            const segs = (analysis && analysis.analytics && analysis.analytics.fight_events) ? analysis.analytics.fight_events : (analysis.segments || []);
            if (this.elems.fightCount) this.elems.fightCount.textContent = segs.length || 0;
            if (this.elems.peopleCount && analysis.analytics && analysis.analytics.people_count_history && analysis.analytics.people_count_history.length) {
                let arr = analysis.analytics.people_count_history.map(x=>x.count);
                this.elems.peopleCount.textContent = Math.round(arr.reduce((a,b)=>a+b,0)/arr.length) || 0;
            }
            if (this.elems.confidence) {
                let best = 0;
                if (segs.length) best = segs.reduce((m,s)=>Math.max(m, s.confidence || s.max_conf || 0), 0);
                this.elems.confidence.textContent = `${Math.round(best)}%`;
            }
            const events = (segs||[]).map(s=>({
                start_time: s.start_time || new Date().toLocaleString(),
                video_time: s.start_sec || (s.start_frame ? (s.start_frame/30) : 0),
                duration: s.duration || (s.end_sec?s.end_sec - s.start_sec:0),
                confidence: s.confidence || s.max_conf || 0,
                repr_frame: s.repr_frame || null
            }));
            this.renderEventLog(events);
            // hotspots
            if (this.elems.hotspotsList) this.fetchHotspots();
        } catch(e){
            console.warn('handleAnalysisResult error', e);
        }
    }

    renderEventLog(events){
        const el = this.elems.eventLog;
        if (!el) return;
        if (!events.length) { el.innerHTML = '<div style="text-align:center;color:#666;padding:20px;">No events</div>'; return; }
        el.innerHTML = events.map(e=>{
            const conf = Math.round((e.confidence||0));
            const dur = e.duration?`${e.duration.toFixed(1)}s`:'—';
            return `<div class="event-item"><div class="event-details">Fight — ${conf}%</div><div class="event-time">${e.start_time} • ${dur}</div></div>`;
        }).join('');
    }

    prependEventLog(ev){
        const el = this.elems.eventLog;
        if (!el) return;
        const prev = el.innerHTML || '';
        const start = ev.start_time || new Date().toLocaleString();
        const dur = ev.duration ? `${ev.duration.toFixed(1)}s` : '—';
        const conf = Math.round((ev.confidence||0));
        const html = `<div class="event-item"><div class="event-details">Fight — ${conf}%</div><div class="event-time">${start} • ${dur}</div></div>`;
        el.innerHTML = html + prev;
    }

    async updateSettings(){
        try {
            const body = {
                body_proximity_threshold: this.elems.bodyThreshold ? parseFloat(this.elems.bodyThreshold.value) : undefined,
                limb_proximity_threshold: this.elems.limbThreshold ? parseFloat(this.elems.limbThreshold.value) : undefined,
                fight_hold_duration: this.elems.holdDuration ? parseInt(this.elems.holdDuration.value) : undefined
            };
            const resp = await fetch('/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
            const j = await this._parseJSON(resp);
            if (j && j.success) this.showAlert('Settings updated','success',2000);
            else this.showAlert('Failed to update settings','error',4000);
        } catch(e) {
            this.showAlert('Settings error','error',4000);
        }
    }

    showAlert(msg, type='info', timeout=4000){
        const el = this.elems.statusAlert;
        if (!el) { console.log(type,msg); return; }
        el.textContent = msg;
        el.className = `alert ${type==='error'?'error':'success'}`;
        el.style.display = 'block';
        if (timeout>0) setTimeout(()=>{ try{ el.style.display='none' } catch(e){} }, timeout);
    }

    async _parseJSON(resp){
        try {
            const ct = (resp.headers && resp.headers.get) ? (resp.headers.get('content-type')||'') : '';
            if (ct.includes('application/json')) return await resp.json();
            const txt = await resp.text();
            try { return JSON.parse(txt); } catch(e){ return {text: txt, ok: resp.ok}; }
        } catch(e){ return null; }
    }
    
    async _updateHazardsUI(hazards) {
        try {
            const el = this.el('hazardsList');
            if (!el) return;
            if (!hazards || Object.keys(hazards).length === 0) {
                el.textContent = 'Нет';
                return;
            }
            const parts = [];
            for (const k of Object.keys(hazards)) {
                const arr = hazards[k] || [];
                if (!arr.length) continue;
                const items = arr.map(d=>`${k.toUpperCase()}: ${Math.round(d.confidence)}%`).join('<br>');
                parts.push(`<div style="margin-bottom:6px;"><strong>${k}</strong><div style="font-size:0.9rem;color:var(--text-secondary)">${items}</div></div>`);
            }
            el.innerHTML = parts.join('');
        } catch(e){
            console.warn('hazards update failed', e);
        }
    }

    async _updateHeatmapAndHotspots() {
        try {
            // heatmap
            const img = this.el('heatmapImg');
            if (img) {
                img.src = '/heatmap?t=' + Date.now();
            }
            // hotspots
            const r = await fetch('/hotspots');
            const j = await this._parseJSON(r);
            if (j && j.success) {
                const list = this.el('hotspotsList');
                if (list) {
                    if (!j.hotspots || !j.hotspots.length) {
                        list.innerHTML = 'Пока пусто';
                    } else {
                        list.innerHTML = j.hotspots.map(h=>`<div style="padding:6px;border-bottom:1px solid var(--border-color);">x:${h.x} y:${h.y} • ${Math.round(h.intensity*100)}% • ${h.events} events</div>`).join('');
                    }
                }
            }
        } catch(e){
            // ignore
        }
    }

    _updateEscalationBanner(escalation) {
        const el = this.el('escalationBanner');
        if (!el) return;
        if (escalation && escalation.active) {
            el.style.display = 'block';
            el.style.color = 'var(--accent-warning)';
            el.textContent = `⚠️ Эскалация: ${escalation.reason || 'Tension detected'} • уровень ${Math.round(escalation.tension_level||0)}`;
        } else {
            el.style.display = 'none';
            el.textContent = '';
        }
    }

    async toggleFaceRecognition(btn) {
        if (!btn) return;

        // Get current state from button text
        const isCurrentlyOn = btn.textContent.includes('ON');
        const newState = !isCurrentlyOn;

        try {
            const resp = await fetch('/toggle_face_recognition', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: newState ? 'true' : 'false'})
            });
            const j = await this._parseJSON(resp);
            if (j && j.success) {
                btn.textContent = `Face Recognition: ${newState ? 'ON' : 'OFF'}`;
                if (newState) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
                this.showAlert(`Face recognition ${newState ? 'enabled' : 'disabled'}`, 'success', 2000);
            } else {
                this.showAlert('Failed to toggle face recognition', 'error', 3000);
            }
        } catch (err) {
            console.error('Error toggling face recognition:', err);
            this.showAlert('Error toggling face recognition', 'error', 3000);
        }
    }

    async toggleFaceBlur(btn) {
        if (!btn) return;

        // Get current state from button text
        const isCurrentlyOn = btn.textContent.includes('ON');
        const newState = !isCurrentlyOn;

        try {
            const resp = await fetch('/toggle_face_blur', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: newState ? 'true' : 'false'})
            });
            const j = await this._parseJSON(resp);
            if (j && j.success) {
                btn.textContent = `Face Blur: ${newState ? 'ON' : 'OFF'}`;
                if (newState) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
                this.showAlert(`Face blur ${newState ? 'enabled' : 'disabled'}`, 'success', 2000);
            } else {
                this.showAlert('Failed to toggle face blur', 'error', 3000);
            }
        } catch (err) {
            console.error('Error toggling face blur:', err);
            this.showAlert('Error toggling face blur', 'error', 3000);
        }
    }

    async loadFeatureStatus() {
        try {
            const resp = await fetch('/feature_status');
            const j = await this._parseJSON(resp);
            if (j && j.success) {
                const faceRecBtn = this.el('faceRecognitionBtn');
                const faceBlurBtn = this.el('faceBlurBtn');

                if (faceRecBtn) {
                    const isEnabled = j.face_recognition_enabled || false;
                    faceRecBtn.textContent = `Face Recognition: ${isEnabled ? 'ON' : 'OFF'}`;
                    if (isEnabled) {
                        faceRecBtn.classList.add('active');
                    } else {
                        faceRecBtn.classList.remove('active');
                    }
                }
                if (faceBlurBtn) {
                    const isEnabled = j.face_blur_enabled || false;
                    faceBlurBtn.textContent = `Face Blur: ${isEnabled ? 'ON' : 'OFF'}`;
                    if (isEnabled) {
                        faceBlurBtn.classList.add('active');
                    } else {
                        faceBlurBtn.classList.remove('active');
                    }
                }
            }
        } catch (err) {
            console.warn('Failed to load feature status:', err);
        }
    }

}

document.addEventListener('DOMContentLoaded', ()=>{ window.fdApp = new FDApp(); });
