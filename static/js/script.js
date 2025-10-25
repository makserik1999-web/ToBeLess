
// static/js/script.js
// Full front-end integration for MJPEG stream + analytics snapshot polling + charts
// Uses Chart.js (already included in index.html)

class FDApp {
    constructor(opts={}) {
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
            confidenceChart: this.el('confidenceChart')
        };

        this.setupUI();
        this.startPolling();
        this._initCharts();
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

        // file change -> show local preview
        if (this.elems.fileInput) {
            this.elems.fileInput.addEventListener('change', (e)=>{
                const f = e.target.files && e.target.files[0];
                if (f) {
                    this.showVideoPreview(f);
                }
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
        if (this.elems.startBtn) { this.elems.startBtn.disabled = true; this.elems.startBtn.innerText = 'Starting…'; }
        try {
            const st = this.elems.sourceType ? this.elems.sourceType.value : '0';
            // Video file upload path -> we will show server-processed stream (/video_feed)
            if (st === 'video') {
                if (!this.elems.fileInput || !this.elems.fileInput.files || this.elems.fileInput.files.length===0) throw new Error('No file selected');
                const fd = new FormData();
                fd.append('file', this.elems.fileInput.files[0]);
                const resp = await fetch('/start_stream', {method:'POST', body: fd});
                const j = await this._parseJSON(resp);
                if (!j || !j.success) throw new Error(j && j.error ? j.error : 'Upload failed');

                this.isAnalyzing = true;
                this.analysisJobId = j.job_id || null;

                // show MJPEG stream (processed frames with skeleton)
                this.hideAllVideoElements();
                if (this.elems.videoFeed) {
                    // add cache-buster
                    this.elems.videoFeed.src = '/video_feed?t=' + Date.now();
                    this.elems.videoFeed.style.display = 'block';
                }
                this.isStreaming = true;
                if (this.elems.stopBtn) { this.elems.stopBtn.disabled = false; }
                this.showAlert('File uploaded — showing processed stream', 'success', 3500);
            } else {
                // start live stream (camera or url)
                const source = (st === '0') ? '0' : (this.elems.sourceInput && this.elems.sourceInput.value) || '';
                const resp = await fetch('/start_stream', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({source})});
                const j = await this._parseJSON(resp);
                if (!j || !j.success) throw new Error(j && j.error ? j.error : 'Start stream failed');

                this.hideAllVideoElements();
                if (this.elems.videoFeed) {
                    this.elems.videoFeed.src = '/video_feed?t=' + Date.now();
                    this.elems.videoFeed.style.display = 'block';
                }
                this.isStreaming = true;
                if (this.elems.stopBtn) { this.elems.stopBtn.disabled = false; }
                this.showAlert('Stream started', 'success', 3000);
            }
        } catch (err) {
            console.error(err);
            this.showAlert('Start failed: '+(err.message||err),'error',8000);
            if (this.elems.stopBtn) this.elems.stopBtn.disabled = true;
        } finally {
            if (this.elems.startBtn) { this.elems.startBtn.disabled = false; this.elems.startBtn.innerText = '▶️ Start Detection'; }
        }
    }

    async stop(){
        try {
            if (this.elems.stopBtn) { this.elems.stopBtn.disabled = true; }
            const resp = await fetch('/stop_stream', {method:'POST'});
            const j = await this._parseJSON(resp);
            if (!j || !j.success) throw new Error(j && j.error ? j.error : 'Stop failed');

            // cleanup
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
                // show final analysis summary
                this.handleAnalysisResult(j.analysis);
                this.showAlert('Background analysis finished', 'success', 4000);
            } else {
                this.showAlert(`Job: ${j.status || 'running'}`, 'info', 1200);
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
        try {
            // priorize job polling when analyzing in background
            if (this.isAnalyzing && this.analysisJobId) {
                await this.pollJobOnce(this.analysisJobId);
                return;
            }
            // always fetch snapshot
            const r = await fetch('/analytics');
            const j = await this._parseJSON(r);
            if (!j || !j.success) return;

            const recent = Array.isArray(j.recent_data) ? j.recent_data : (j.recent_data || []);
            const latest = (recent && recent.length) ? recent[recent.length - 1] : null;

            // robust derive people
            let people = 0;
            if (latest && (latest.people !== undefined)) {
                people = latest.people;
            } else if (j.analytics && Array.isArray(j.analytics.people_count_history) && j.analytics.people_count_history.length) {
                const lastEntry = j.analytics.people_count_history[j.analytics.people_count_history.length - 1];
                people = lastEntry && lastEntry.count !== undefined ? lastEntry.count : 0;
            } else if (j.current_status && j.current_status.people !== undefined) {
                people = j.current_status.people;
            }

            let fight = false;
            if (latest && latest.fight !== undefined) fight = !!latest.fight;
            else if (j.current_status && j.current_status.fight_detected !== undefined) fight = !!j.current_status.fight_detected;

            let conf = 0;
            if (latest && latest.metrics && latest.metrics.confidence !== undefined) conf = latest.metrics.confidence;
            else if (j.analytics && j.analytics.detection_confidence_history && j.analytics.detection_confidence_history.length) {
                conf = j.analytics.detection_confidence_history[j.analytics.detection_confidence_history.length - 1].confidence || 0;
            }

            if (this.elems.peopleCount) this.elems.peopleCount.textContent = people;
            if (this.elems.fightCount) this.elems.fightCount.textContent = fight ? 1 : 0;
            if (this.elems.confidence) this.elems.confidence.textContent = `${Math.round(conf)}%`;
            if (this.elems.fps) this.elems.fps.textContent = Math.round(30);

            if (this.elems.statusIndicator) {
                if (fight) {
                    this.elems.statusIndicator.className = 'status-indicator status-fight';
                    this.elems.statusIndicator.textContent = ' Fight detected';
                } else {
                    this.elems.statusIndicator.className = 'status-indicator status-normal';
                    this.elems.statusIndicator.textContent = ' System Ready';
                }
            }

            // chart
            this.appendChart(people, Math.round(conf));

            // if server finished streaming and returned analytics -> show final results once
            if (!j.streaming && j.analytics) {
                if (!this.isAnalyzing) {
                    try { this.handleAnalysisResult({ analytics: j.analytics }); } catch(e){ console.warn('final handle failed', e); }
                }
            }

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
                if (segs.length) {
                    best = segs.reduce((m,s)=>Math.max(m, s.confidence || s.max_conf || 0), 0);
                }
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
            this.renderSegmentsGallery(events);
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

    renderSegmentsGallery(events){
        const el = this.elems.segmentsGallery;
        if (!el) return;
        if (!events.length) { el.innerHTML = '<div style="text-align:center;color:#666;padding:12px;">No segments</div>'; return; }
        el.innerHTML = events.map(e=>{
            const thumb = e.repr_frame ? e.repr_frame : '';
            const conf = Math.round((e.confidence||0));
            const t = e.video_time!=null?`${e.video_time.toFixed(1)}s`:'—';
            return `<div class="segment-card">${thumb?`<a href="${thumb}" target="_blank"><img src="${thumb}" alt="frame" /></a>`:`<div class="no-thumb">No preview</div>`}<div class="seg-meta">t=${t} • ${conf}%</div></div>`;
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
        el.className = `alert ${type}`;
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
}

document.addEventListener('DOMContentLoaded', ()=>{ window.fdApp = new FDApp(); });

