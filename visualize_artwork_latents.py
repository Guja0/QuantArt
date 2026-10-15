"""Compare artwork-encoder latents across baseline and available fine-tuned models.

Writes an offline interactive HTML explorer, common PCA/t-SNE projections,
cluster contact sheets, a PNG overview, numeric embeddings, and provenance.
Clusters are appearance groups, not verified painting-medium labels.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/quantart-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from plotly.offline import get_plotlyjs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import torch

from finetune_artwork_decoder import ENCODER_PREFIXES, load_state_dict, pick_device, to_tensor
from taming.modules.diffusionmodules.model import Encoder
from omegaconf import OmegaConf

COLORS = ['#2563eb', '#e6550d', '#16a34a', '#9333ea', '#d22f6e', '#0891b2', '#8b6f47', '#737373']


def fingerprint(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare_image(path, size):
    with Image.open(path) as im:
        original = ImageOps.exif_transpose(im).convert('RGB')
    fitted = ImageOps.contain(original, (size, size), Image.Resampling.LANCZOS)
    w, h = fitted.size
    left, top = (size - w) // 2, (size - h) // 2
    pixels = np.pad(np.asarray(fitted), ((top, size-h-top), (left, size-w-left), (0, 0)), mode='edge')
    # Valid-region mask excludes edge padding from pooled statistics and Gram descriptors.
    mask = np.zeros((size, size), dtype=np.float32)
    mask[top:top+h, left:left+w] = 1
    thumb = ImageOps.contain(original, (128, 128), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format='JPEG', quality=85)
    return to_tensor(Image.fromarray(pixels)), torch.from_numpy(mask)[None], thumb, base64.b64encode(buffer.getvalue()).decode()


def discover_models(root, base):
    models = [dict(id='baseline', label='Baseline coco2art', state=base, run=None, weights=None, epochs=0)]
    for args_file in sorted(root.glob('*/args.json')):
        args = json.loads(args_file.read_text())
        history_file = args_file.parent / 'history.json'
        if not history_file.exists():
            continue
        history = json.loads(history_file.read_text())
        epochs = history['history'][-1]['epoch']
        part = args.get('train_part', 'decoder')
        suffix = 'autoencoder' if part == 'both' else ('encoder' if part == 'encoder' else 'decoder')
        chosen = [('last', epochs)]
        if history.get('best_epoch', epochs) != epochs:
            chosen.append(('best', history['best_epoch']))
        if part in ('encoder', 'both'):
            chosen.append(('last_raw', epochs))
        for tag, epoch in chosen:
            filename = f'{tag}_{suffix}.pt' if tag != 'last_raw' else f'last_{suffix}_raw.pt'
            weights = args_file.parent / 'checkpoints' / filename
            if not weights.exists():
                continue
            state = torch.load(weights, map_location='cpu', weights_only=True)
            # All current experiments share the same coco2art pretrained checkpoint.
            source = Path(args['ckpt']).resolve()
            if source != Path(OPTIONS.ckpt).resolve():
                raise ValueError(f'{args_file.parent.name} uses a different pretrained checkpoint: {source}')
            updated = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIXES)}
            encoder = updated or base
            if set(encoder) != set(base):
                raise ValueError(f'Incomplete encoder state: {weights}')
            for key in base:
                if base[key].shape != encoder[key].shape:
                    raise ValueError(f'Encoder architecture differs: {weights}')
            if part in ('encoder', 'both') and not updated:
                raise ValueError(f'Encoder checkpoint has no encoder weights: {weights}')
            label = f'{args_file.parent.name} / {tag} / epoch {epoch}'
            models.append(dict(id=args_file.parent.name+'_'+tag, label=label, state=encoder,
                               run=str(args_file.parent), weights=str(weights), epochs=epoch, part=part))
    return models


def extract(model, projection, images, masks, device, batch_size):
    descriptors = {'statistics': [], 'gram': []}
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = images[start:start+batch_size].to(device)
            z = projection(model(x))
            mask = torch.nn.functional.interpolate(masks[start:start+batch_size].to(device), size=z.shape[-2:], mode='area').flatten(2)
            count = mask.sum(-1).clamp_min(1)
            flat = z.flatten(2)
            mean = (flat * mask).sum(-1) / count
            variance = ((flat - mean[..., None]).square() * mask).sum(-1) / count
            descriptors['statistics'].append(torch.cat([mean, variance.clamp_min(0).sqrt()], dim=1).cpu().numpy())
            weighted = flat * mask.sqrt()
            gram = weighted @ weighted.transpose(1, 2) / count[..., None]
            triangle = torch.triu_indices(z.shape[1], z.shape[1], device=device)
            descriptors['gram'].append(gram[:, triangle[0], triangle[1]].cpu().numpy())
    return {key: normalize(np.concatenate(values)).astype(np.float32) for key, values in descriptors.items()}


def make_contact_sheet(out, key, model_id, labels, features, centers, thumbs, names):
    sheet = Image.new('RGB', (6*150, len(centers)*178), 'white')
    draw = ImageDraw.Draw(sheet)
    for c in range(len(centers)):
        members = np.flatnonzero(labels == c)
        draw.text((8, c*178+5), f'Cluster {c+1} | {len(members)} artworks', fill=COLORS[c])
        if not len(members):
            continue
        distances = np.linalg.norm(features[members] - centers[c], axis=1)
        chosen = members[np.argsort(distances)[:6]]
        for j, index in enumerate(chosen):
            thumb = thumbs[index]
            x, y = j*150, c*178+24
            sheet.paste(thumb, (x+(150-thumb.width)//2, y+(128-thumb.height)//2))
            draw.text((x+4, y+130), Path(names[index]).stem, fill='black')
    path = out / 'clusters' / f'{key}_{model_id}.jpg'
    path.parent.mkdir(exist_ok=True)
    sheet.save(path, quality=93)


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>QuantArt latent-space explorer</title>
<style>body{font-family:system-ui,sans-serif;background:#f4f6fa;color:#172033;margin:0;padding:24px}h1{margin:0 0 8px;font-size:26px}.note{max-width:1100px;line-height:1.5}select,button{padding:8px;border:1px solid #ccd2dc;border-radius:6px;background:white}label{display:inline-block;margin:8px 14px 8px 0}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}.plot{height:520px;background:white;border-radius:12px}.gallery{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}.card{background:white;padding:10px;border-radius:8px;width:140px;cursor:pointer}.card img{width:128px;height:128px;object-fit:contain}.card p{margin:3px 0;font-size:12px}.selected{border:2px solid #172033}#details{margin:14px 0;padding:12px;background:white;border-radius:8px}#status{padding:8px 0}#clusterSummary{display:flex;gap:14px;flex-wrap:wrap}@media(max-width:1000px){.plots{grid-template-columns:1fr}}</style>
<script>__PLOTLY__</script></head><body><h1>QuantArt artwork latent space</h1>
<p class="note">180 artworks, processed identically for every model. Decoder-only runs reuse the unchanged baseline encoder, so their latent maps are identical. Colors are six unsupervised appearance groups anchored to the baseline encoder, not oil/acrylic/brushstroke labels. Palette, subject, frames, lighting and texture can all affect proximity.</p>
<label>Descriptor <select id="feature"><option value="gram">Latent Gram correlations (appearance / texture)</option><option value="statistics">Latent channel means and standard deviations</option></select></label>
<label>Projection <select id="projection"><option value="pca">Shared PCA</option><option value="tsne">Joint t-SNE</option></select></label>
<label>Left model <select id="left"></select></label><label>Right model <select id="right"></select></label>
<label>Thumbnail overlay <input type="checkbox" id="thumbnails"></label>
<p class="note">Both panels use a common projection and axis scale. Click any point or thumbnail to compare its nearest neighbors in the original descriptor space. The 2D map is an approximation; t-SNE gaps and cluster sizes are not quantitative distances.</p>
<div id="status"></div><div class="plots"><div id="leftPlot" class="plot"></div><div id="rightPlot" class="plot"></div></div>
<div id="details"></div><div id="neighbors"></div>
<h2>Inspect an appearance group</h2><label>Group <select id="cluster"><option value="all">All artworks</option></select></label><div id="clusterSummary"></div><div id="gallery" class="gallery"></div>
<script>
const data=__DATA__,colors=__COLORS__;let selected=0;
const $=id=>document.getElementById(id),models=data.models;
for(const m of models){for(const id of ['left','right']){const o=document.createElement('option');o.value=m.id;o.textContent=m.label;$(id).appendChild(o)}}
$('left').value='baseline';$('right').value=data.defaultModel;
for(let i=0;i<data.clusters;i++){const o=document.createElement('option');o.value=i;o.textContent='Cluster '+(i+1);$('cluster').appendChild(o)}
const model=id=>models.find(m=>m.id===id);
const view=id=>data.features[$('feature').value].groups[model(id).group];
function card(index,subtitle=''){const d=document.createElement('div');d.className='card'+(index===selected?' selected':'');const img=document.createElement('img');img.src='data:image/jpeg;base64,'+data.thumbnails[index];d.appendChild(img);for(const t of [data.names[index],subtitle]){if(t){const p=document.createElement('p');p.textContent=t;d.appendChild(p)}}d.onclick=()=>{selected=index;render()};return d}
function plot(side){const m=model($(side).value),g=view(m.id),pts=g[$('projection').value],projection=data.features[$('feature').value].bounds[$('projection').value];
let trace={type:'scatter',mode:'markers',x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),customdata:data.names.map((_,i)=>i),text:data.names.map((n,i)=>n+' · cluster '+(g.labels[i]+1)),hovertemplate:'%{text}<extra></extra>',marker:{size:data.names.map((_,i)=>i===selected?15:8),color:g.labels.map(c=>colors[c]),line:{width:1,color:'white'},opacity:0.9}};
const images=$('thumbnails').checked?pts.map((p,i)=>({source:'data:image/jpeg;base64,'+data.thumbnails[i],xref:'x',yref:'y',x:p[0],y:p[1],sizex:(projection.x[1]-projection.x[0])*0.035,sizey:(projection.y[1]-projection.y[0])*0.05,xanchor:'center',yanchor:'middle',layer:'below',opacity:0.9})):[];
Plotly.react(side+'Plot',[trace],{title:{text:m.label,font:{size:13}},margin:{t:55,l:45,b:40,r:15},xaxis:{range:projection.x,title:{text:$('projection').value==='pca'?'PC1':'t-SNE 1'}},yaxis:{range:projection.y,title:{text:$('projection').value==='pca'?'PC2':'t-SNE 2'}},images,paper_bgcolor:'white',plot_bgcolor:'white'},{responsive:true,displaylogo:false}).then(()=>{const el=$(side+'Plot');el.removeAllListeners('plotly_click');el.on('plotly_click',event=>{selected=event.points[0].customdata;render()})})}
function render(){plot('left');plot('right');const l=model($('left').value),r=model($('right').value),g=view(r.id),f=data.features[$('feature').value];
$('status').textContent=l.group===r.group?'These models have identical artwork encoder weights: the same coordinates and neighbors are expected.':'These models have different artwork encoder weights. Compare neighbor lists and images, rather than judging only 2D separation.';
$('details').textContent=data.names[selected]+' · Left cluster '+(view(l.id).labels[selected]+1)+' · Right cluster '+(g.labels[selected]+1)+' · Baseline six-group silhouette '+f.silhouette.toFixed(3)+' (chosen group count, not a medium-classification score).';
$('neighbors').replaceChildren();for(const m of [l,r]){const section=document.createElement('section'),title=document.createElement('h3');title.textContent='Nearest artworks — '+m.label;section.appendChild(title);const row=document.createElement('div');row.className='gallery';row.appendChild(card(selected,'Selected artwork'));view(m.id).neighbors[selected].forEach((n,j)=>row.appendChild(card(n,'Neighbor '+(j+1))));section.appendChild(row);$('neighbors').appendChild(section)}
$('clusterSummary').replaceChildren();for(let c=0;c<data.clusters;c++){const s=document.createElement('span');s.style.color=colors[c];s.textContent='Cluster '+(c+1)+': '+g.labels.filter(x=>x===c).length;$('clusterSummary').appendChild(s)}
$('gallery').replaceChildren();data.names.forEach((_,i)=>{if($('cluster').value==='all'||g.labels[i]===Number($('cluster').value))$('gallery').appendChild(card(i,'Cluster '+(g.labels[i]+1)))})}
for(const id of ['feature','projection','left','right','thumbnails','cluster'])$(id).addEventListener('change',render);render();
</script></body></html>'''


def main():
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--runs', default='outputs/finetune_decoder')
    parser.add_argument('--out', default='outputs/latent_visualization')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--clusters', type=int, default=6)
    OPTIONS = parser.parse_args()
    if OPTIONS.size < 32 or OPTIONS.size % 16 or not 2 <= OPTIONS.clusters <= len(COLORS):
        parser.error('Use a size divisible by 16 and 2–8 clusters')
    torch.set_num_threads(4)
    device = pick_device(OPTIONS.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    out = Path(OPTIONS.out)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.relative_to(OPTIONS.data).as_posix() for p in Path(OPTIONS.data).rglob('*') if p.is_file() and not p.name.startswith('._') and '__MACOSX' not in p.parts and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    if len(names) < 10:
        raise ValueError('Need at least ten artworks for this visualization')
    prepared = [prepare_image(Path(OPTIONS.data)/name, OPTIONS.size) for name in names]
    images = torch.stack([p[0] for p in prepared])
    masks = torch.stack([p[1] for p in prepared])
    thumbs, encoded = [p[2] for p in prepared], [p[3] for p in prepared]
    p = OmegaConf.load(OPTIONS.config).model.params
    state = load_state_dict(OPTIONS.ckpt)
    base = {k: v for k,v in state.items() if k.startswith(ENCODER_PREFIXES)}
    del state
    models = discover_models(Path(OPTIONS.runs), base)
    encoder, projection = Encoder(**p.ddconfig).to(device).eval(), torch.nn.Conv2d(p.ddconfig.z_channels,p.embed_dim,1).to(device).eval()
    unique, extracted, metadata = {}, {}, []
    for m in models:
        digest = fingerprint(m['state'])
        if digest not in unique:
            group = 'encoder_'+str(len(unique))
            unique[digest] = group
            encoder.load_state_dict({k[len('encoder_real.'):]:v for k,v in m['state'].items() if k.startswith('encoder_real.')}, strict=True)
            projection.load_state_dict({k[len('quant_conv_real.'):]:v for k,v in m['state'].items() if k.startswith('quant_conv_real.')},strict=True)
            print('Extracting',m['label'], 'for', len(names), 'artworks', flush=True)
            extracted[group] = extract(encoder, projection, images, masks, device, OPTIONS.batch_size)
        m['group'] = unique[digest]
        item = {k:v for k,v in m.items() if k!='state'}
        item['encoder_sha256'] = digest
        metadata.append(item)
    print('Models:',len(models),'distinct artwork encoders:',len(extracted),flush=True)
    if not all(np.isfinite(x).all() for desc in extracted.values() for x in desc.values()):
        raise RuntimeError('Nonfinite latent descriptors')
    features, numeric, summary = {}, {}, {}
    for key in ['statistics','gram']:
        print('Projecting',key,flush=True)
        groups = list(extracted)
        pooled = np.concatenate([extracted[g][key] for g in groups])
        pca = PCA(n_components=min(50,len(pooled)-1,pooled.shape[1]),svd_solver='randomized',random_state=23)
        reduced = pca.fit_transform(pooled)
        joint = TSNE(n_components=2,perplexity=min(30,(len(pooled)-1)/3),random_state=23,init='pca',learning_rate='auto',max_iter=1000,n_jobs=4).fit_transform(reduced)
        cluster = KMeans(n_clusters=OPTIONS.clusters,random_state=23,n_init=20).fit(reduced[:len(names)])
        features[key] = {'groups':{}, 'silhouette':float(silhouette_score(reduced[:len(names)],cluster.labels_)),'bounds':{}}
        for method, xy in [('pca',reduced[:,:2]),('tsne',joint)]:
            pad=(xy.max(0)-xy.min(0))*0.07+1e-8
            features[key]['bounds'][method]={'x':[float(xy[:,0].min()-pad[0]),float(xy[:,0].max()+pad[0])],'y':[float(xy[:,1].min()-pad[1]),float(xy[:,1].max()+pad[1])]}
        base_neighbors = None
        summary[key] = {'pca_variance_first_two':float(pca.explained_variance_ratio_[:2].sum()),'pca_variance_first_50':float(pca.explained_variance_ratio_.sum()),'baseline_silhouette':features[key]['silhouette'],'groups':{}}
        for gi,g in enumerate(groups):
            values = extracted[g][key]
            sl=slice(gi*len(names),(gi+1)*len(names))
            labels = cluster.predict(reduced[sl])
            distances = np.clip(1-values@values.T,0,2)
            np.fill_diagonal(distances,np.inf)
            neighbors = np.argsort(distances,axis=1)[:,:8]
            if base_neighbors is None: base_neighbors=neighbors
            overlap = np.mean([len(set(a)&set(b))/8 for a,b in zip(base_neighbors,neighbors)])
            features[key]['groups'][g]={'pca':reduced[sl,:2].tolist(),'tsne':joint[sl].tolist(),'labels':labels.tolist(),'neighbors':neighbors.tolist()}
            numeric[f'{key}_{g}_descriptors']=values
            numeric[f'{key}_{g}_pca']=reduced[sl,:2]
            numeric[f'{key}_{g}_tsne']=joint[sl]
            summary[key]['groups'][g]={'neighbor_overlap_with_baseline':float(overlap),'cluster_counts':np.bincount(labels,minlength=OPTIONS.clusters).tolist()}
            make_contact_sheet(out,key,g,labels,reduced[sl],cluster.cluster_centers_,thumbs,names)
    default=next((m['id'] for m in metadata if m.get('part')=='encoder' and m['id'].endswith('_last')),metadata[-1]['id'])
    payload={'names':names,'thumbnails':encoded,'models':metadata,'features':features,'clusters':OPTIONS.clusters,'defaultModel':default}
    html=HTML.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload).replace('</','<\\/')).replace('__COLORS__',json.dumps(COLORS))
    (out/'latent_explorer.html').write_text(html)
    np.savez_compressed(out/'latent_descriptors_and_coordinates.npz',**numeric)
    report={'settings':vars(OPTIONS),'image_count':len(names),'models':metadata,'distinct_encoders':len(extracted),'analysis':summary,'notes':['Descriptors use pre-codebook artwork-encoder latents; decoder-only changes cannot alter them.','Images are resized and edge-padded to 256px; pooling masks padding, but convolution boundary effects remain.','All unique encoders share one fitted PCA and one joint t-SNE per descriptor.','Clusters are fitted on baseline PCA-50 features and assigned to other encoders with those same centers.','Gram correlations and pooled statistics are heuristic appearance descriptors; no medium or brushstroke labels are available.','No claim that fine-tuning improves semantic clustering; neighbors are computed from original normalized descriptors.','All artwork images are exploratory, including train/validation/test and near-duplicates.','Museum photos, wall labels, composition, palette, lighting and frames can influence groups.']}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    with (out/'coordinates_and_neighbors.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['model','descriptor','image','cluster','pca_x','pca_y','tsne_x','tsne_y','nearest_images'])
        for m in metadata:
            for key in features:
                view=features[key]['groups'][m['group']]
                for i,name in enumerate(names):
                    writer.writerow([m['label'],key,name,view['labels'][i]+1,*view['pca'][i],*view['tsne'][i],';'.join(names[j] for j in view['neighbors'][i])])
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,m in zip(axes,[metadata[0],next(m for m in metadata if m['id']==default)]):
        view=features['gram']['groups'][m['group']]
        xy=np.array(view['pca'])
        for c in range(OPTIONS.clusters):
            ids=np.flatnonzero(np.array(view['labels'])==c)
            ax.scatter(xy[ids,0],xy[ids,1],s=24,c=COLORS[c],label=f'Group {c+1} (n={len(ids)})',alpha=0.85)
        ax.set_title(m['label'],fontsize=10)
        bounds=features['gram']['bounds']['pca']
        ax.set_xlim(bounds['x']);ax.set_ylim(bounds['y']);ax.set_xlabel('Shared PC1');ax.set_ylabel('Shared PC2');ax.legend(fontsize=8)
    fig.suptitle('Artwork latent Gram appearance descriptors — baseline vs encoder fine-tuning')
    fig.savefig(out/'baseline_vs_encoder_pca.png',dpi=180)
    plt.close(fig)
    print('Saved interactive explorer:',out/'latent_explorer.html',flush=True)
    print(json.dumps({'models':len(metadata),'distinct_encoders':len(extracted),'neighbor_overlap_gram':summary['gram']['groups']},indent=2),flush=True)


if __name__ == '__main__':
    main()
