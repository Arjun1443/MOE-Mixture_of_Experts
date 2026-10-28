import numpy as np
import pandas as pd
import streamlit as st
from ui import page

page('Inside a Mixture of Experts','See how a sparse router selects experts, combines their outputs, and distributes tokens.','MODEL ARCHITECTURE')
with st.sidebar:
    experts=st.slider('Expert networks',2,12,8)
    top_k=st.slider('Experts per token',1,experts,min(2,experts))
    tokens=st.slider('Simulated tokens',8,128,32)
    seed=st.number_input('Random seed',value=42,min_value=0,max_value=9999)
rng=np.random.default_rng(seed)
logits=rng.normal(size=(tokens,experts))
indices=np.argsort(logits,axis=1)[:,-top_k:]
weights=np.zeros_like(logits)
selected=np.take_along_axis(logits,indices,axis=1)
p=np.exp(selected-selected.max(axis=1,keepdims=True)); p/=p.sum(axis=1,keepdims=True)
np.put_along_axis(weights,indices,p,axis=1)
loads=(weights>0).sum(axis=0)
a,b,c=st.columns(3)
a.metric('Total experts',experts)
b.metric('Active per token',top_k)
c.metric('Routing sparsity',f'{1-top_k/experts:.0%}')
router,load,method=st.tabs(['Token routing','Expert utilization','Architecture notes'])
with router:
    t=st.slider('Inspect token',0,tokens-1,0)
    st.bar_chart(pd.DataFrame({'Expert':[f'E{i}' for i in range(experts)],'Mixture weight':weights[t]}).set_index('Expert'))
    st.caption('Top-k logits are selected, then normalized with a softmax over selected experts only.')
    expert_outputs=rng.normal(size=(experts,4))
    combined=weights[t]@expert_outputs
    st.code('output(token) = Σ routing_weight(expert) × expert_output(token)')
    st.dataframe(pd.DataFrame({'Output dimension':range(4),'Weighted output':combined}),hide_index=True)
with load:
    st.bar_chart(pd.DataFrame({'Expert':[f'E{i}' for i in range(experts)],'Assigned tokens':loads}).set_index('Expert'))
    st.download_button('Download routing weights',pd.DataFrame(weights).to_csv(index=False),'routing.csv','text/csv')
with method:
    st.markdown("""### An interactive architecture simulation
The router logits and expert outputs are seeded synthetic values. No language model is trained or queried here. Change top-k to explore the sparsity–utilization tradeoff.

The notebook contains the PyTorch language-model implementation and references AviSoori1x/makeMoE. This companion app explains its routing concept; it does not claim to reproduce trained-model performance.
""")
