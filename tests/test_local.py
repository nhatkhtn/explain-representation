import torch



def test_approximator_class():
    from exrep.model import LocalRepresentationApproximator
    
    device = "cuda"
    approximator = LocalRepresentationApproximator(4, 5, 6, 1.0)
    query = torch.randn(10, 4, device=device)
    key = torch.randn(11, 5, device=device)
    logits = approximator(query, key)
    encoded_query = approximator.encode(query=query)
    encoded_key = approximator.encode(key=key)
    eq, ek = approximator.encode(query=query, key=key)
    assert torch.allclose(eq, encoded_query) and torch.allclose(ek, encoded_key), "Encoding mismatch"
    assert logits.shape == (10, 11), f"Expected logits to have shape (10, 11), got {logits.shape}"
    print("Logits shape =", logits.shape)
    print("Encoded query shape =", encoded_query.shape)
    print("Encoded key shape =", encoded_key.shape)

def test_training():
    from exrep.train import train_local_representation

    device = "cuda"

    model_config = dict(
        output_dim=5,
        temperature=0.21,
    )
    loss_config = dict()

    optimizer_config = dict(
        lr=1e-3,
        weight_decay=1e-2,
    )

    query_inputs = torch.randn(10, 3)
    query_targets = torch.randn(10, 8)
    keys = torch.randn(6, 8)

    train_local_representation(
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        query_inputs=query_inputs,
        query_targets=query_targets,
        keys=keys,
        groups=[(0, 1), 2],
        alpha=1.0,
        num_epochs=2,
        batch_size=3,
        log_every_n_steps=1,
        device=device,   
    )