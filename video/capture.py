"""Takes real screenshots of the live site and app for the intro video, with a headless Edge controlled over the DevTools protocol.
Uses a throw-away browser profile, so nothing of the author's own data or sessions is involved; the app shows its clearly labelled demo data."""
import base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
import websocket

EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shots'); os.makedirs(OUT, exist_ok=True)
PORT = 9333
SITE = os.environ.get('SITE', 'https://wetlandear.tech')
APP = os.environ.get('APP', 'https://app.wetlandear.tech')


class CDP:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix='we-shots-')
        self.proc = subprocess.Popen([EDGE, '--headless=new', '--disable-gpu', '--no-first-run', '--hide-scrollbars', '--mute-audio', '--remote-allow-origins=*', '--remote-debugging-port=%d' % PORT,
                                      '--user-data-dir=' + self.tmp, '--window-size=1440,900', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen('http://127.0.0.1:%d/json' % PORT, timeout=2)); break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError('browser did not start')
        page = [t for t in tabs if t.get('type') == 'page'][0]
        self.ws = websocket.create_connection(page['webSocketDebuggerUrl'], timeout=60, max_size=None); self.n = 0

    def call(self, method, **params):
        self.n += 1; i = self.n; self.ws.send(json.dumps({'id': i, 'method': method, 'params': params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get('id') == i:
                if 'error' in m: raise RuntimeError('%s: %s' % (method, m['error']))
                return m.get('result', {})

    def ev(self, js, wait=False):
        r = self.call('Runtime.evaluate', expression=js, awaitPromise=wait, returnByValue=True)
        return (r.get('result') or {}).get('value')

    def viewport(self, w, h, dsf=1, mobile=False):
        self.call('Emulation.setDeviceMetricsOverride', width=w, height=h, deviceScaleFactor=dsf, mobile=mobile)

    def go(self, url, wait=4.0):
        self.call('Page.navigate', url=url); time.sleep(wait)

    def shot(self, name, full=False):
        path = os.path.join(OUT, name + '.png')
        r = self.call('Page.captureScreenshot', format='png', captureBeyondViewport=bool(full))
        open(path, 'wb').write(base64.b64decode(r['data'])); print('shot', name, os.path.getsize(path) // 1024, 'KB', flush=True)

    def close(self):
        try: self.ws.close()
        except Exception: pass
        self.proc.terminate()
        try: self.proc.wait(10)
        except Exception: self.proc.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)


def main():
    b = CDP()
    try:
        # ---- the project site as a visitor sees it: viewport-sized frames while scrolling (blocks reveal themselves on scroll)
        b.viewport(1440, 900, 1); b.go(SITE, 4)
        h = int(b.ev('document.documentElement.scrollHeight') or 900)
        n_frames = min(7, max(2, h // 800))
        for i in range(n_frames):
            y = i * 780
            for k in range(1, 5):   # glide to the position so that scroll-triggered animations fire
                b.ev('window.scrollTo(0,%d)' % int(y * k / 4)); time.sleep(0.35)
            time.sleep(1.3); b.shot('site_%d' % i)
        print('site frames:', n_frames, flush=True)

        if os.environ.get('SITE_ONLY') == '1': return
        # ---- the app on a phone, with the clearly labelled demo data
        b.viewport(390, 844, 2, True); b.go(APP + '/#pulse', 5)
        b.ev("try{localStorage.setItem('we.lang','ru')}catch(e){}")
        b.ev("(function(){WEStore.seedDemo(); WEApp.show('pulse'); var s=document.getElementById('signBanner'); if(s) s.hidden=true;})()"); time.sleep(2.5); b.shot('app_pulse')
        b.ev("WEQuick.open()"); time.sleep(1.2); b.shot('app_quick'); b.ev("WEQuick.close()"); time.sleep(0.6)
        b.ev("WEApp.show('science'); setTimeout(function(){var s=document.getElementById('indSug'); if(s) s.click();},400)"); time.sleep(2); b.shot('app_science')
        b.ev("WEApp.show('calendar')"); time.sleep(2); b.shot('app_calendar')
        b.ev("WEApp.show('map')"); time.sleep(5); b.shot('app_map')
        b.ev("WEApp.show('log')"); time.sleep(2); b.shot('app_log')
        b.ev("WEApp.show('manage')"); time.sleep(2); b.shot('app_manage')
        # ---- desktop view of the app
        b.viewport(1440, 900, 1); b.go(APP + '/#pulse', 4)
        b.ev("(function(){WEStore.seedDemo(); WEApp.show('pulse'); var s=document.getElementById('signBanner'); if(s) s.hidden=true;})()"); time.sleep(2.5); b.shot('app_desktop')
    finally:
        b.close()


if __name__ == '__main__':
    main()
