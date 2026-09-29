"""Entry point for data generation scripts module."""

# This allows running the stats script with: python -m data.generation.scripts

if __name__ == "__main__":
    # Import the stats module
    from data.generation.scripts.stats import main

    main()
