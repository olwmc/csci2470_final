## Assumptions
- Different intermediate layers have meaningfully different activation spaces -- they encode different stuff at different layers.
#### Mixture model
Try and train `num_layers` separate small models that can act as "experts" at their respective layer, then collect them into a mixture model.

Maybe we could learn independent, layerwise siamese networks? And then pass it into `CosineWeighter`? I think weighted layerwise cosine distance is a really good way to go, but it's just that our activation spaces aren't really good for that. Maybe we can learn a new representation that is more amenable to cosine similarity?

#### Loss function
* Do we just try to get the model to output a scalar? Or do we discretize a scalar into a set of categories? (I.e. instead of just [0,1] the model outputs a 6 dimensional vector representing logits for [0/5, 1/5, 2/5, 3/5, 4/5, 5/5]). Getting the model to output a scalar seems like it would be hard to train. But there's stuff like https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html that make that more possible.

* Maybe we can somehow figure out how to incorporate correlation into the loss function

### The Dumbest Possible Model
In [[Findings#10/23/24]] we saw that cosine similarity weakly correlates with human similarity. Here's a dumb idea. Maybe the model can learn to weight the cosines at each layer. I.e. something like
```python
class CosineWeighter(nn.Module):
    def __init__(self, n_layers=24, d_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(n_layers, d_hidden)
        self.out = nn.Linear(d_hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.sigmoid(self.out(h))
        
        return h.squeeze()

cw = CosineWeighter().to('cuda:0')

optimizer = torch.optim.AdamW(cw.parameters(), lr=3e-4)
loss_fn = torch.nn.MSELoss()

def epoch(model, sim_data, scores, batch_size=32):
    model.train()

    n_batches = sim_data.shape[0] // batch_size

    for batch in range(n_batches):
        data_batch   = sim_data[batch*batch_size: (batch+1) * batch_size].to('cuda:0')
        scores_batch = torch.Tensor(scores[batch*batch_size: (batch+1) * batch_size]).to('cuda:0')

        optimizer.zero_grad()

        outputs = model(data_batch)
        loss = loss_fn(outputs, scores_batch)

        loss.backward()

        optimizer.step()

    return loss.item()

for _ in tqdm(range(100)):
    epoch(cw, train_cosines, train_scores)

print(torch.corrcoef(torch.stack([cw.to('cpu')(test_cosines), test_scores])))
print(torch.corrcoef(torch.stack([cw.to('cpu')(train_cosines), train_scores])))
```
This code gives us:
```
tensor([[1.0000, 0.3824],
        [0.3824, 1.0000]], grad_fn=<ClampBackward1>)
tensor([[1.0000, 0.3680],
        [0.3680, 1.0000]], grad_fn=<ClampBackward1>)      
```
So 38% train correlation, 36% test. Already better than cosine!

So now okay maybe instead of just doing raw cosine, we learn a transformation, do cosine on that, and then do everything else the same. Maybe learning a transformation of activation space would be useful here. Ideally it would learn a different transformation for each layer. Like a set of layerwise siamese networks. Maybe we start off training the siamese networks like encoders, get a good reconstruction loss, and then swap them over to being siamese networks. Then we can leverage the goodness of training an autoencoder (learns a good representation that we don't have to just use )