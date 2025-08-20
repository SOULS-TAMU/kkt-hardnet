def run_explicit_model_instance():
    # ------------------- SYMBOLIC SETUP -------------------
    x1, x2, b = symbols('x1 x2 b')
    x_syms = [x1, x2]
    residuals = Matrix([x1 ** 2 + x2 - b, x1 + x2 ** 2 - b])

    # Solve for x1, x2 in terms of b
    solutions = solve(residuals, x_syms, dict=True)

    # Pick only real solutions with simple form (e.g., avoid complex branches)
    sol_fn = lambdify(b, [solutions[0][x1], solutions[0][x2]], modules='numpy')

    # Dataset Creation and Model Running
    dataset = BDataset(sol_fn=sol_fn)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NewtonModel(residuals, x_syms, [b]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    trainer = KKT_HardNet_Trainer(model, train_loader, val_loader, test_loader, optimizer, criterion,
                                  num_epochs=700, eta=1e-3, model_loss_tolerance=1e-6)
    trainer.train()