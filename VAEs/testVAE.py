import VAE

encoder,decoder = VAE.generateBackbones(4)
vae = VAE.VAE(encoder,decoder)
print(vae)