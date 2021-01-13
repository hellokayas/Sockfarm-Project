# Sockfarm-Project

## create conda env from yaml
If you want to create a new conda env, do the follows
```bash
conda env create -f environment.yaml
```

If you want to modify the existing env, do the follows instead
```
conda env update
```

## overall organization

```bash
├── Data
├── LICENSE
├── README.md
├── resources
├── rev2data
└── src
```

## rev2data
Here's the structure of rev2data.
```bash
rev2data
├── alpha
│   ├── alpha_gt.csv
│   └── alpha_network.csv
├── amazon
│   ├── amazon_gt.csv
│   └── amazon_network.csv
├── epinions
│   ├── epinions_gt.csv
│   └── epinions_network.csv
└── otc
    ├── otc_gt.csv
    └── otc_network.csv
```
