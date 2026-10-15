"""Compare original, decoder-only, encoder-only, and joint artwork autoencoders.

Identical deterministic inputs, final-epoch checkpoints for the main comparison,
plus best-checkpoint and older-run controls. No training or photo-transfer occurs.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLCONFIGDIR','/tmp/quantart-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from finetune_artwork_decoder import ContinuousArtworkAE, ENCODER_PREFIXES, DECODER_PREFIXES, edge_loss, load_lpips, pick_device, resize_shorter, to_image, to_tensor

MAIN = ['original','decoder_only','encoder_only','joint']
VIEWS = ['whole512','center256','detail256']
METRICS = ['l1','psnr','ssim','lpips','edge_error','laplacian_error','sharpness_ratio','sharpness_gap']
UP = {'psnr','ssim'}


def prepare(im):
    whole = ImageOps.contain(im,(512,512),Image.Resampling.LANCZOS)
    w,h=whole.size;left,top=(512-w)//2,(512-h)//2
    pixels=np.pad(np.asarray(whole),((top,512-h-top),(left,512-w-left),(0,0)),mode='edge')
    center=resize_shorter(im,256);cw,ch=center.size
    center_box=((cw-256)//2,(ch-256)//2,(cw+256)//2,(ch+256)//2)
    enlarged=resize_shorter(im,min(640,min(im.size)))
    ew,eh=enlarged.size
    best=None
    # Select the most textured of nine deterministic target-only patches.
    for y in np.linspace(0,eh-256,3).round().astype(int):
        for x in np.linspace(0,ew-256,3).round().astype(int):
            patch=enlarged.crop((int(x),int(y),int(x)+256,int(y)+256))
            a=np.asarray(patch,dtype=np.float32)/255
            score=float(np.abs(np.diff(a,axis=0)).mean()+np.abs(np.diff(a,axis=1)).mean())
            if best is None or score>best[0]:best=(score,(int(x),int(y),int(x)+256,int(y)+256),patch)
    return {'whole512':(Image.fromarray(pixels),(left,top,left+w,top+h)),
            'center256':(center.crop(center_box),(0,0,256,256)),
            'detail256':(best[2],(0,0,256,256))}, {'source_size':im.size,'whole_content_box':(left,top,left+w,top+h),'detail_resized_size':enlarged.size,'detail_box':best[1],'detail_target_gradient':best[0]}


def laplacian(x):
    k=x.new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]]).view(1,1,3,3)
    return F.conv2d(x.mean(1,keepdim=True),k)


def ssim(a,b):
    a,b=(a+1)/2,(b+1)/2
    axis=torch.arange(11,device=a.device,dtype=a.dtype)-5
    g=torch.exp(-axis.square()/(2*1.5**2));g=g/g.sum()
    kernel=(g[:,None]*g[None,:]).view(1,1,11,11).repeat(3,1,1,1)
    ma=F.conv2d(a,kernel,groups=3);mb=F.conv2d(b,kernel,groups=3)
    va=F.conv2d(a*a,kernel,groups=3)-ma.square()
    vb=F.conv2d(b*b,kernel,groups=3)-mb.square()
    cov=F.conv2d(a*b,kernel,groups=3)-ma*mb
    return (((2*ma*mb+0.01**2)*(2*cov+0.03**2))/((ma.square()+mb.square()+0.01**2)*(va+vb+0.03**2))).mean().item()


def scores(pred,target,lpips):
    pred=pred.clamp(-1,1);target=target.clamp(-1,1)
    mse=((pred-target)/2).square().mean().clamp_min(1e-12)
    la,lb=laplacian(pred),laplacian(target)
    output_energy,target_energy=la.abs().mean(),lb.abs().mean()
    sharp=1.0 if target_energy.item()<1e-6 and output_energy.item()<1e-6 else (output_energy/target_energy.clamp_min(1e-6)).item()
    return {'l1':F.l1_loss(pred,target).item(),'psnr':(-10*torch.log10(mse)).item(),
            'ssim':ssim(pred,target),'lpips':lpips(pred.contiguous(),target.contiguous()).mean().item(),
            'edge_error':edge_loss(pred,target).mean().item(),'laplacian_error':F.l1_loss(la,lb).item(),
            'sharpness_ratio':sharp,'sharpness_gap':abs(sharp-1)}


def overlay(model,state,prefixes):
    expected={k for k in model.state_dict() if k.startswith(prefixes)}
    if set(state)!=expected:raise ValueError('Checkpoint must contain exactly the expected artwork autoencoder modules')
    model.load_state_dict(state,strict=False)


def variants(root):
    specs=[('decoder_only','dev_coco2art_60ep','last','decoder'),
           ('encoder_only','encoder_coco2art_60ep','last','encoder'),
           ('joint','joint_coco2art_edge_60ep','last','autoencoder'),
           ('encoder_best','encoder_coco2art_60ep','best','encoder'),
           ('joint_best','joint_coco2art_edge_60ep','best','autoencoder'),
           ('decoder_gan','decoder_coco2art_gan_60ep_20261014','last','decoder'),
           ('decoder_gan_edge','decoder_coco2art_gan_edge_60ep_20261014','last','decoder'),
           ('decoder_all87','cu_art_coco2art_100ep','last','decoder')]
    result=[{'id':'original','label':'Original coco2art','epoch':0,'views':VIEWS,'run':None,'weights':None,'part':'none'}]
    labels={'decoder_only':'Decoder-only, epoch 60','encoder_only':'Encoder-only, epoch 60 (frozen tuned decoder)',
            'joint':'Joint encoder+decoder, epoch 60','encoder_best':'Best encoder, epoch 55 (frozen tuned decoder)',
            'joint_best':'Best joint autoencoder, epoch 59','decoder_gan':'Decoder + GAN, epoch 60',
            'decoder_gan_edge':'Decoder + GAN + edge, epoch 60','decoder_all87':'Decoder all-images, epoch 87'}
    for id,run,tag,part in specs:
        folder=root/run
        a=json.loads((folder/'args.json').read_text());h=json.loads((folder/'history.json').read_text())
        epoch=h['best_epoch'] if tag=='best' else h['history'][-1]['epoch']
        if id in MAIN and epoch!=60:raise ValueError(f'{run} has not completed 60 epochs')
        weight=folder/'checkpoints'/f'{tag}_{part}.pt'
        if not weight.exists():raise FileNotFoundError(weight)
        result.append({'id':id,'label':labels[id],'epoch':epoch,'views':VIEWS if id in MAIN else ['center256'],
                       'run':str(folder),'weights':str(weight),'part':part,'settings':a,
                       'training_images':180 if a.get('all_images') else 140,
                       'frozen_decoder':str(folder/'checkpoints/frozen_decoder.pt') if part=='encoder' else None})
    return result


def parameter_changes(base,state):
    result={}
    for side,prefix in [('encoder',ENCODER_PREFIXES),('decoder',DECODER_PREFIXES)]:
        sq,total,n=0.,0.,0
        changed=0
        for k in base:
            if k.startswith(prefix):
                b=base[k];a=state[k].detach().cpu()
                sq+=(a-b).square().sum().item();total+=b.square().sum().item();n+=b.numel()
                changed+=int(not torch.equal(a,b))
        result[side]={'changed_tensors':changed,'parameter_count':n,'rms_change':float(np.sqrt(sq/n)),
                      'relative_l2_change':float(np.sqrt(sq/max(total,1e-20)))}
    return result


def labeled_strip(images,labels,height=300):
    cellw=330;header=44
    canvas=Image.new('RGB',(cellw*len(images),height+header),'white');draw=ImageDraw.Draw(canvas)
    for j,(im,label) in enumerate(zip(images,labels)):
        thumb=ImageOps.contain(im,(cellw-8,height-8),Image.Resampling.LANCZOS)
        canvas.paste(thumb,(j*cellw+(cellw-thumb.width)//2,header+(height-thumb.height)//2))
        draw.text((j*cellw+6,4),label,fill='black')
    return canvas


HTML=r'''<!doctype html><html><head><meta charset="utf-8"><title>QuantArt all-artwork comparison</title><style>
body{font:15px system-ui;margin:24px;background:#f4f6fa;color:#182033}h1{font-size:27px}select,input,button{padding:8px;margin:6px;border-radius:5px;border:1px solid #b7beca;background:white}table{border-collapse:collapse;background:white;width:100%;font-size:12px}td,th{padding:8px;border:1px solid #dfe3e9;text-align:right}td:first-child,th:first-child{text-align:left}.comparison{width:100%;background:white;margin-top:12px}.comparison img{width:100%;height:auto}.gallery{display:flex;gap:10px;flex-wrap:wrap}.card{background:white;border-radius:8px;padding:8px;cursor:pointer;width:128px}.card img{width:128px;height:110px;object-fit:contain}.card p{font-size:12px;margin:2px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.pair img{width:100%;max-height:600px;object-fit:contain;background:white}.note{line-height:1.5;max-width:1100px}#meta{padding:12px;background:white;border-radius:8px}@media(max-width:900px){.pair{grid-template-columns:1fr}}</style></head><body>
<h1>Original vs fine-tuned artwork autoencoders</h1><p class="note">Identical deterministic inputs across 180 artworks. Main comparison: final epoch-60 EMA checkpoints, with no training augmentation. The encoder-only experiment froze an already fine-tuned decoder: compare encoder-only against decoder-only to isolate its encoder update. Joint training also used edge loss, so these runs are not a controlled training-part ablation. Reconstructions only; photo stylization is a separate test.</p>
<label>View <select id="view"><option value="whole512">Whole artwork, up to 512 px</option><option value="center256">Training-scale center crop, 256 px</option><option value="detail256">Most-textured target patch, 256 px</option></select></label><label>Subset <select id="split"><option value="all">All 180</option><option value="train">Train 140</option><option value="val">Validation 20</option><option value="test">Held-out test 20</option></select></label>
<h2>Metric summary</h2><p class="note">Lower: L1, LPIPS, edge error, Laplacian error, sharpness gap. Higher: PSNR and SSIM. Sharpness ratio near 1 matches target edge energy; a larger ratio can mean noise or halos. All metrics use display-clamped predictions. Decoder-all87 trained on all images, including the nominal test subset.</p><div id="summary"></div>
<h2>Inspect every painting</h2><label>Image <select id="image"></select></label><button id="previous">Previous</button><button id="next">Next</button><label>Heat maps <input type="checkbox" id="heat"></label>
<div id="meta"></div><div id="strip" class="comparison"></div><div id="imageMetrics"></div>
<h3>Full-resolution side-by-side inspection</h3><label>Left <select id="left"></select></label><label>Right <select id="right"></select></label><div class="pair"><div><h4 id="leftTitle"></h4><a id="leftLink"><img id="leftImage"></a></div><div><h4 id="rightTitle"></h4><a id="rightLink"><img id="rightImage"></a></div></div>
<h3>Image gallery</h3><input id="search" placeholder="Filter filenames"><div id="gallery" class="gallery"></div>
<script>const DATA=__DATA__;const $=id=>document.getElementById(id);const models=DATA.models,metrics=DATA.metrics;let index=0;
const available=()=>models.filter(m=>m.views.includes($('view').value));const subset=()=>DATA.images.filter(i=>$('split').value==='all'||i.split===$('split').value);
function table(rows){const t=document.createElement('table'),h=document.createElement('tr');for(const text of ['Model',...metrics]){const th=document.createElement('th');th.textContent=text;h.appendChild(th)}t.appendChild(h);for(const row of rows){const tr=document.createElement('tr');for(const value of [row.label,...metrics.map(k=>row[k])]){const td=document.createElement('td');td.textContent=typeof value==='number'?value.toFixed(4):value;tr.appendChild(td)}t.appendChild(tr)}return t}
function options(){const ids=subset();$('image').replaceChildren();for(const im of ids){const o=document.createElement('option');o.value=im.index;o.textContent=im.name;$('image').appendChild(o)}if(!ids.some(im=>im.index===index))index=ids[0].index;$('image').value=index;
for(const id of ['left','right']){const old=$(id).value;$(id).replaceChildren();for(const m of available()){const o=document.createElement('option');o.value=m.id;o.textContent=m.label;$(id).appendChild(o)}$(id).value=available().some(m=>m.id===old)?old:(id==='left'?'original':'joint')}}
function render(){const view=$('view').value,split=$('split').value,im=DATA.images[index],heat=$('heat').checked;$('summary').replaceChildren(table(available().map(m=>({label:m.label,...DATA.aggregate[view][split][m.id]}))));$('meta').textContent=im.name+' · '+im.split+' · Original size '+im.source_size.join(' × ')+' · Detail crop selected from the input image only, identically for all models.';
const img=document.createElement('img');img.src='comparisons/'+view+'/'+im.id+(heat?'_errors.jpg':'.jpg');$('strip').replaceChildren(img);$('imageMetrics').replaceChildren(table(available().map(m=>({label:m.label,...DATA.rows[view][m.id][index]}))));
for(const side of ['left','right']){const m=models.find(m=>m.id===$(side).value);$(side+'Title').textContent=m.label;const path=(heat?'errors':'reconstructions')+'/'+view+'/'+m.id+'/'+im.id+(heat?'.jpg':'.png');$(side+'Image').src=path;$(side+'Link').href=path;$(side+'Link').target='_blank'}
$('gallery').replaceChildren();const query=$('search').value.toLowerCase();for(const item of subset().filter(x=>x.name.toLowerCase().includes(query))){const card=document.createElement('div');card.className='card';const thumb=document.createElement('img');thumb.src='inputs/'+view+'/'+item.id+'.png';card.appendChild(thumb);const text=document.createElement('p');text.textContent=item.name;card.appendChild(text);card.onclick=()=>{index=item.index;$('image').value=index;render()};$('gallery').appendChild(card)}}
for(const id of ['view','split'])$(id).onchange=()=>{options();render()};for(const id of ['left','right','heat','search'])$(id).onchange=render;$('image').onchange=()=>{index=Number($('image').value);render()};for(const [id,delta] of [['previous',-1],['next',1]])$(id).onclick=()=>{const ids=subset();let at=ids.findIndex(im=>im.index===index);index=ids[(at+delta+ids.length)%ids.length].index;$('image').value=index;render()};options();render();
</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/coco2art.yaml')
    parser.add_argument('--ckpt',default='../checkpoints-20261013T103406Z-1-001/checkpoints/last.ckpt')
    parser.add_argument('--data',default='../datasets/CU_data_jpg')
    parser.add_argument('--split',default='datasets/CU_data_split')
    parser.add_argument('--runs',default='outputs/finetune_decoder')
    parser.add_argument('--out',default='outputs/model_comparison_all_artworks')
    parser.add_argument('--device',default='cuda');parser.add_argument('--batch_size',type=int,default=2)
    opt=parser.parse_args();torch.set_num_threads(4);device=pick_device(opt.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=Path(opt.out)
    if (out/'summary.json').exists():raise ValueError('Completed comparison already exists; choose a new --out')
    out.mkdir(parents=True,exist_ok=True)
    names=sorted(p.relative_to(opt.data).as_posix() for p in Path(opt.data).rglob('*') if p.is_file() and not p.name.startswith('._') and '__MACOSX' not in p.parts and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    splits={}
    for split in ['train','val','test']:
        for name in Path(opt.split,split+'.txt').read_text().splitlines():
            if name in splits:raise ValueError('Overlapping dataset splits')
            splits[name]=split
    images=[];inputs={v:[] for v in VIEWS};boxes={v:[] for v in VIEWS}
    for i,name in enumerate(names):
        with Image.open(Path(opt.data)/name) as im:im=ImageOps.exif_transpose(im).convert('RGB')
        views,info=prepare(im);id=f'{i:03d}_'+re.sub(r'[^A-Za-z0-9_.-]','_',Path(name).stem)
        images.append({'index':i,'id':id,'name':name,'split':splits.get(name,'unlisted'),**info})
        for view,(pic,box) in views.items():
            inputs[view].append(to_tensor(pic));boxes[view].append(box)
            path=out/'inputs'/view/(id+'.png');path.parent.mkdir(parents=True,exist_ok=True)
            pic.crop(box).save(path)
    inputs={k:torch.stack(v) for k,v in inputs.items()}
    cases=variants(Path(opt.runs));net=ContinuousArtworkAE.from_checkpoint(opt.config,opt.ckpt)
    base={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};net=net.to(device).eval().requires_grad_(False)
    lpips=load_lpips(device);rows=[];data_rows={v:{} for v in VIEWS};changes={}
    with torch.inference_mode():
        for case in cases:
            print('Model:',case['label'],flush=True);net.load_state_dict(base,strict=True)
            if case['id']!='original':
                a=case['settings']
                if Path(a['ckpt']).resolve()!=Path(opt.ckpt).resolve():raise ValueError('Different pretrained checkpoint across runs')
                if case['part']=='encoder':
                    frozen=torch.load(case['frozen_decoder'],map_location='cpu',weights_only=True)
                    overlay(net,frozen,DECODER_PREFIXES)
                    tuned=torch.load(case['weights'],map_location='cpu',weights_only=True)
                    overlay(net,tuned,ENCODER_PREFIXES)
                else:
                    tuned=torch.load(case['weights'],map_location='cpu',weights_only=True)
                    overlay(net,tuned,ENCODER_PREFIXES+DECODER_PREFIXES if case['part']=='autoencoder' else DECODER_PREFIXES)
                del tuned
                case['weights_sha256']=hashlib.sha256(Path(case['weights']).read_bytes()).hexdigest()
            changes[case['id']]=parameter_changes(base,net.state_dict())
            for view in case['views']:
                print('  View:',view,flush=True);data_rows[view][case['id']]=[]
                for start in range(0,len(names),opt.batch_size):
                    x=inputs[view][start:start+opt.batch_size].to(device)
                    pred=net.decode(net.encode(x))
                    if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite reconstruction')
                    for j in range(len(x)):
                        i=start+j;left,top,right,bottom=boxes[view][i]
                        target=x[j:j+1,:,top:bottom,left:right]
                        output=pred[j:j+1,:,top:bottom,left:right].clamp(-1,1)
                        metric=scores(output,target,lpips)
                        row={'image':names[i],'split':images[i]['split'],'view':view,'model':case['id'],**metric}
                        rows.append(row);data_rows[view][case['id']].append(metric)
                        folder=out/'reconstructions'/view/case['id'];folder.mkdir(parents=True,exist_ok=True)
                        to_image(output[0]).save(folder/(images[i]['id']+'.png'))
                        error=(output[0]-target[0]).abs().mean(0).cpu().numpy()/2
                        heat=(plt.get_cmap('magma')(np.clip(error/0.25,0,1))[...,:3]*255).astype(np.uint8)
                        folder=out/'errors'/view/case['id'];folder.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(heat).save(folder/(images[i]['id']+'.jpg'),quality=90)
                    if (start+opt.batch_size)%40==0:print('   ',min(start+opt.batch_size,len(names)),'/',len(names),flush=True)
    aggregate={v:{} for v in VIEWS};deltas=[]
    for view in VIEWS:
        for split in ['all','train','val','test']:
            aggregate[view][split]={}
            for case in cases:
                chosen=[r for r in rows if r['view']==view and r['model']==case['id'] and (split=='all' or r['split']==split)]
                if not chosen:continue
                average={k:float(np.mean([r[k] for r in chosen])) for k in METRICS};average['count']=len(chosen)
                aggregate[view][split][case['id']]=average
                base_average=aggregate[view][split]['original']
                deltas.append({'view':view,'split':split,'model':case['id'],**average,**{'delta_'+k:average[k]-base_average[k] for k in METRICS}})
    for filename,values in [('per_image_metrics.csv',rows),('aggregate_metrics.csv',deltas)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    for view in VIEWS:
        folder=out/'comparisons'/view;folder.mkdir(parents=True,exist_ok=True)
        for im in images:
            path=im['id'];pics=[];heats=[]
            with Image.open(out/'inputs'/view/(path+'.png')) as pic:target=pic.copy()
            pics.append(target);heats.append(target)
            for id in MAIN:
                with Image.open(out/'reconstructions'/view/id/(path+'.png')) as pic:pics.append(pic.copy())
                with Image.open(out/'errors'/view/id/(path+'.jpg')) as pic:heats.append(pic.copy())
            labels=['Input','Original','Decoder only','Encoder only + tuned decoder','Joint encoder + decoder']
            labeled_strip(pics,labels,height=400 if view=='whole512' else 280).save(folder/(path+'.jpg'),quality=94)
            labeled_strip(heats,labels,height=280).save(folder/(path+'_errors.jpg'),quality=92)
    report={'settings':vars(opt),'images':images,'models':cases,'parameter_changes':changes,'metrics':METRICS,'aggregate':aggregate,
            'notes':['Main checkpoints are final epoch 60 EMA weights, not best-selected weights.','Encoder-only uses the stored fine-tuned frozen decoder; decoder-only vs encoder-only isolates the encoder update.','Joint training uses edge weight 0.2; decoder-only and encoder-only use edge weight 0, so architecture causality cannot be isolated.','Primary train/validation/test split is 140/20/20. The secondary decoder-all87 trained on all 180.','whole512 preserves the full composition using edge padding; output padding is removed before metrics.','center256 reproduces training-scale deterministic center cropping. detail256 selects the highest target-gradient patch among nine candidates after resizing the shorter side to min(640,native).','All image outputs and metrics are display-clamped to [-1,1]. RGB L1 uses that range; PSNR/SSIM use [0,1].','Error maps use a fixed RGB-MAE scale from 0 to 0.25 in [0,1] for every model.','Sharpness ratio measures edge energy, not whether invented texture is correct; use target-matching edge/Laplacian errors and actual crops.','No claim about photo style-transfer quality; these tests assess reconstruction.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    payload={'images':images,'models':cases,'metrics':METRICS,'aggregate':aggregate,'rows':data_rows}
    (out/'comparison.html').write_text(HTML.replace('__DATA__',json.dumps(payload).replace('</','<\\/')))
    # Paired improvements and report tables for the held-out set.
    md=['# Original vs fine-tuned artwork autoencoders','',f'Compared **{len(names)} artworks**. Main models use final epoch-60 EMA weights.','',
        '## Important interpretation','',
        '- Encoder-only training froze a decoder already fine-tuned for 60 epochs. Its reconstruction changes relative to original therefore include that decoder. Compare encoder-only to decoder-only to isolate the encoder update.',
        '- Joint training uses edge loss 0.2, while the non-GAN single-side experiments used edge loss 0. These are real experiment comparisons, not a controlled architecture ablation.',
        '- Train 140 / validation 20 / test 20 are reported separately. The older decoder-all87 control trained on all 180; its test-subset scores are in-sample.',
        '- Medium/brushstroke labels are unavailable. Metrics measure reconstruction agreement; visual crops and error maps show missing or invented detail.','',
        '## Main held-out test results','']
    for view in VIEWS:
        md += [f'### {view}','', '| Model | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Edge error ↓ | Laplacian error ↓ | Sharpness ratio ≈ 1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for case in cases:
            if case['id'] in MAIN:
                a=aggregate[view]['test'][case['id']]
                md.append('| '+case['label']+' | '+' | '.join(f'{a[k]:.4f}' for k in METRICS[:-1])+' |')
        md.append('')
    md += ['## What changed in the weights','', '| Model | Encoder relative L2 change | Decoder relative L2 change |','|---|---:|---:|']
    for case in cases:
        c=changes[case['id']];md.append(f"| {case['label']} | {c['encoder']['relative_l2_change']:.6f} | {c['decoder']['relative_l2_change']:.6f} |")
    md += ['','## Paired reconstruction changes on all images','']
    for case in cases:
        if case['id'] not in MAIN or case['id']=='original':continue
        for view in VIEWS:
            now=data_rows[view][case['id']];old=data_rows[view]['original']
            win=sum(a['lpips']<b['lpips'] for a,b in zip(now,old));edge_win=sum(a['edge_error']<b['edge_error'] for a,b in zip(now,old))
            md.append(f"- {case['label']}, {view}: lower LPIPS for {win}/{len(names)} artworks; lower target edge error for {edge_win}/{len(names)}.")
    md += ['','## Best checkpoints and previous GAN experiments','',
        'The interactive table and aggregate CSV include the best encoder (epoch 55), best joint model (epoch 59), previous decoder + GAN, decoder + GAN + edge, and the older all-images decoder (epoch 87), evaluated on identical center256 crops. These supplementary checkpoints have separate labels; they are not substituted into the main epoch-60 comparison.','',
        '## Files','', '- `comparison.html`: offline interactive explorer; select views/subsets/models, inspect every painting, compare pixels and error maps.',
        '- `comparisons/`: 540 labeled input/original/decoder/encoder/joint strips and matching error-map strips.',
        '- `inputs/`, `reconstructions/`, `errors/`: individual full-image/crop PNGs and fixed-scale error maps.',
        '- `per_image_metrics.csv`: every image/model/view metric.', '- `aggregate_metrics.csv`: averages and signed differences from the original model for each subset.',
        '- `summary.json`: preprocessing, exact checkpoint paths/hashes, settings, and weight changes.',
        '- `metrics_overview.png`: primary-model test-set metrics.', '- `tensorboard/`: aggregate metrics and representative image comparisons.',
        '', 'Higher PSNR or lower LPIPS alone does not establish sharper painting texture. The reported sharpness ratio can reward noise, while target edge/Laplacian errors penalize disagreement. Refer to brush-detail comparisons before choosing a model.']
    (out/'RESULTS.md').write_text('\n'.join(md)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,metric in zip(axes.flat,['lpips','psnr','ssim','edge_error','laplacian_error','sharpness_ratio']):
        for j,view in enumerate(VIEWS):
            values=[aggregate[view]['test'][id][metric] for id in MAIN]
            ax.bar(np.arange(4)+(j-1)*0.24,values,width=0.24,label=view)
        ax.set_xticks(np.arange(4),['Original','Decoder','Encoder*','Joint']);ax.set_title(metric)
        if metric=='sharpness_ratio':ax.axhline(1,c='black',ls='--')
        ax.legend(fontsize=7)
    fig.suptitle('Held-out test artwork reconstruction — *encoder uses previously tuned frozen decoder')
    fig.savefig(out/'metrics_overview.png',dpi=160);plt.close(fig)
    writer=SummaryWriter(str(out/'tensorboard'))
    for view in VIEWS:
        for split,table in aggregate[view].items():
            for id,values in table.items():
                for metric in METRICS:writer.add_scalar(f'{view}/{split}/{metric}/{id}',values[metric],60 if id!='original' else 0)
        for im in images[:6]:
            with Image.open(out/'comparisons'/view/(im['id']+'.jpg')) as pic:
                writer.add_image(f'comparisons/{view}/{im["name"]}',np.asarray(pic),60,dataformats='HWC')
    writer.close()
    print('COMPLETED:',out.resolve(),flush=True)
    print(json.dumps(aggregate['center256']['test'],indent=2),flush=True)


if __name__=='__main__':main()
