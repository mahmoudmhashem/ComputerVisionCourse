from nn import *

def main():
    model = Sequential(
        Layer(3, 4),
        Relu(),
        Layer(4, 4),
        Relu(),
        Layer(4, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.05)



    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, 0.0, 0.0, 1.0] # desired targets

    for k in range(1000):
    
        # forward pass
        ypreds = []
        for x, y in zip(xs, ys):
            ypred = model(x)[0]
            ypreds.append(ypred)

        loss:Value = criterion(ys, ypreds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss}")

    print()
    print("Predictions:")

    for ypred in ypreds:
        print(ypred)


if __name__ == "__main__":
    main()