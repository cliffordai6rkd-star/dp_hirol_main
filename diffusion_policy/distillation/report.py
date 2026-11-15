"""Small post-validation curves; never invoked in a training microbatch."""
import json
from pathlib import Path


def learning_curves(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(directory)
    points=sorted((json.loads(p.read_text()) for p in root.glob('validation_*.json')
                   if p.stem.rsplit('_',1)[-1].isdigit()),key=lambda x:x['step'])
    if not points:return
    figure,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for phase,color in [('contact','#B24A32'),('precontact','#C59728'),('free','#37768A')]:
        valid=[p for p in points if phase in p['summary'].get('internal',{})]
        steps=[p['step'] for p in valid]
        data=[p['summary']['internal'][phase] for p in valid]
        for ax,key in zip(axes.flat,['kd','enhanced_sampled_action_mse','rgb_rms','actual_response_mse']):
            values=[d.get(key,float('nan')) for d in data]
            ax.plot(steps,values,marker='o',label=phase,color=color)
            ax.set_xlabel('optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax,title in zip(axes.flat,['FP32 denoiser KD (episode macro)','Complete native sampled action MSE',
                                  'RGB RMS (episode macro)','Actual visual response MSE']):
        ax.set_title(title,fontsize=10)
    figure.savefig(root/'learning_curves.png',dpi=150);plt.close(figure)
