"""Pilot: normalized market snapshot -> rule-based analysis -> PDF -> Telegram.

No live data provider or LLM is configured. Never present demo values as real.
"""
import argparse
import io
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image


def analyse(item, now):
    """Consume CLOSED one-hour bars only; stale/invalid inputs cannot emit trades."""
    out = dict(name=item['name'], symbol=item['symbol'], investment='INSUFFICIENT DATA',
               news=item.get('news', []), trading='INSUFFICIENT DATA', reason='Missing or invalid bars')
    try:
        d = pd.DataFrame(item['bars'])
        d['time'] = pd.to_datetime(d['time'], utc=True)
        d = d[d.time <= pd.Timestamp(now)].sort_values('time')
        if d.time.duplicated().any() or len(d) < 60:
            return out
        cols = ['open', 'high', 'low', 'close', 'volume']
        d[cols] = d[cols].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(d[cols].to_numpy()).all():
            return out
        if (d[['open','high','low','close']] <= 0).any().any() or (d.volume < 0).any():
            return out
        if ((d.high < d[['open','close','low']].max(axis=1)) |
            (d.low > d[['open','close','high']].min(axis=1))).any():
            return out
        out['asof'] = d.time.iloc[-1].isoformat()
        if now - d.time.iloc[-1].to_pydatetime() > timedelta(minutes=100):
            out['reason'] = 'Stale data / market closed. No new signal.'
            return out
        c = d.close
        ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        tr = pd.concat([d.high-d.low, (d.high-c.shift()).abs(), (d.low-c.shift()).abs()],axis=1).max(axis=1)
        atr = tr.tail(14).mean()
        delta = c.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        rsi = 50 if gain == loss == 0 else (100 if loss == 0 else 100-100/(1+gain/loss))
        prior = d.iloc[-21:-1]
        resistance, support = float(prior.high.max()), float(prior.low.min())
        price = float(c.iloc[-1])
        baseline = prior.volume.mean()
        volume_ratio = float(d.volume.iloc[-1]/baseline) if baseline > 0 else 0
        trend = 'UP' if price > ema20 > ema50 else ('DOWN' if price < ema20 < ema50 else 'SIDEWAYS')
        buy = trend == 'UP' and price > resistance and volume_ratio >= 1.5 and rsi < 80
        sell = trend == 'DOWN' and price < support and volume_ratio >= 1.5
        signal = 'BUY' if buy else ('SELL' if sell else 'HOLD / WAIT')
        stop = price-2*atr
        out.update(trading=signal, reason='Rule-based pilot; not backtested. News not scored.',
                   trend=trend, price=price, support=support, resistance=resistance,
                   rsi=float(rsi), volume_ratio=volume_ratio,
                   entry=[price,price+0.25*atr] if buy else None,
                   stop=float(stop) if buy and stop > 0 else None,
                   targets=[float(price+4*atr),float(price+6*atr)] if buy and stop > 0 else [],
                   bars=d.tail(60).to_dict('records'))
        # Fundamental valuation is deliberately not inferred from a chart.
        return out
    except (KeyError, ValueError, TypeError):
        return out


def make_pdf(rows, path, demo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = getSampleStyleSheet()
    story = [Paragraph('MARKET WATCH | '+('DEMO - SYNTHETIC DATA' if demo else 'PILOT'), styles['Title']),
             Spacer(1,18), Paragraph('Hourly trading screen / 6-12 month investment watch',styles['Heading2']),
             Paragraph('This prototype validates report delivery. It does not yet include a live data feed, AI news analysis or fundamental valuation. Signals are illustrative rules, not validated recommendations.', styles['BodyText']), Spacer(1,18)]
    table = [['Stock','Trading','6-12 months']]+[[r['symbol'],r['trading'],r['investment']] for r in rows]
    t = Table(table, colWidths=[150,220,330], repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#132b46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf3f8'),colors.white])]))
    story.append(t)
    for r in rows:
        story += [PageBreak(), Paragraph(escape(r['name'])+' | '+escape(r['symbol']),styles['Title']),
                  Paragraph(escape(r['trading'])+' | Investment: '+escape(r['investment']),styles['Heading2']),
                  Paragraph(escape(r['reason']),styles['BodyText'])]
        if 'price' in r:
            story.append(Paragraph(f"Price: {r['price']:.2f} | Support: {r['support']:.2f} | Resistance: {r['resistance']:.2f} | RSI: {r['rsi']:.1f}",styles['BodyText']))
            story.append(Paragraph(f"Trend: {r['trend']} | Volume ratio: {r['volume_ratio']:.2f} | Closed bar: {r['asof']}",styles['BodyText']))
            if r['entry']:
                story.append(Paragraph(f"Entry: {r['entry'][0]:.2f}-{r['entry'][1]:.2f} | Stop: {r['stop']:.2f} | Targets: {r['targets'][0]:.2f}, {r['targets'][1]:.2f}",styles['BodyText']))
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot([b['close'] for b in r['bars']],color='#2463a6',linewidth=2)
            ax.axhline(r['support'],color='#b84a48',linestyle='--',label='Support')
            ax.axhline(r['resistance'],color='#358265',linestyle='--',label='Resistance')
            ax.set_xlabel('Closed hourly bars'); ax.set_ylabel('Price'); ax.legend(loc='upper left'); fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf,format='png',dpi=130); plt.close(fig); buf.seek(0)
            story += [Spacer(1,10),Image(buf,width=700,height=210)]
        story.append(Paragraph('News / sources',styles['Heading3']))
        for n in r['news'][:4]:
            story.append(Paragraph(escape(str(n.get('title','')))+'<br/>'+escape(str(n.get('published_at','')))+' | '+escape(str(n.get('url',''))),styles['BodyText']))
        if not r['news']:
            story.append(Paragraph('No connected news source. This does not mean no news occurred.',styles['BodyText']))
    def footer(canvas, doc):
        canvas.setFont('Helvetica',8)
        canvas.drawString(35,20,('SYNTHETIC DEMO - NOT LIVE MARKET DATA' if demo else 'PILOT - data quality and model limits apply'))
        canvas.drawRightString(805,20,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=landscape(A4),leftMargin=40,rightMargin=40,topMargin=30,bottomMargin=35).build(story,onFirstPage=footer,onLaterPages=footer)


def demo_snapshot():
    now = datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)
    rng = np.random.default_rng(12)
    items=[]
    for name, symbol, base in [('Demonstration A','DEMO-A',40),('Demonstration B','DEMO-B',100)]:
        bars=[]
        for i in range(80):
            p=base+i*0.08+float(rng.normal(0,0.1))
            bars.append(dict(time=(now-timedelta(hours=79-i)).isoformat(),open=p-0.05,high=p+0.3,low=p-0.3,close=p,volume=1000))
        if symbol=='DEMO-A':
            p=bars[-2]['high']+0.5
            bars[-1].update(open=p-0.1,high=p+0.2,low=p-0.2,close=p,volume=2200)
        items.append(dict(name=name,symbol=symbol,bars=bars,news=[]))
    return dict(demo=True,generated_at=now.isoformat(),stocks=items)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input'); p.add_argument('--demo',action='store_true'); p.add_argument('--send',action='store_true'); p.add_argument('--output',default='output')
    args=p.parse_args()
    if not args.demo and not args.input:
        p.error('Provide --input normalized snapshot.json, or --demo')
    data=demo_snapshot() if args.demo else json.loads(Path(args.input).read_text())
    if args.send and (args.demo or data.get('demo')):
        p.error('Demo reports cannot be sent by this program')
    now=datetime.now(timezone.utc)
    rows=[analyse(s,now) for s in data['stocks']]
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    stamp=now.strftime('%Y%m%dT%H%M%SZ')
    pdf=out/f'report-{stamp}.pdf'
    make_pdf(rows,pdf,args.demo or data.get('demo',False))
    clean=[{k:v for k,v in r.items() if k!='bars'} for r in rows]
    (out/f'report-{stamp}.json').write_text(json.dumps(clean,indent=2,ensure_ascii=False))
    if args.send:
        import requests
        token=os.environ['TELEGRAM_BOT_TOKEN']; chat=os.environ['TELEGRAM_CHAT_ID']
        # Never log request URLs: Telegram embeds the secret in the URL.
        try:
            with pdf.open('rb') as stream:
                res=requests.post(f'https://api.telegram.org/bot{token}/sendDocument',data={'chat_id':chat,'caption':'Market Watch | Hourly pilot report'},files={'document':stream},timeout=60)
            if res.status_code!=200 or not res.json().get('ok'):
                raise SystemExit('Telegram rejected report; check credentials and group permissions.')
        except requests.RequestException:
            raise SystemExit('Telegram network failure; no automatic retry to avoid duplicates.') from None
    print(f'Created {pdf}')


if __name__=='__main__':
    main()
