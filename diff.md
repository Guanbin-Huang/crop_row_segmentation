# 1.
optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

# 2.
one epoch  one update
