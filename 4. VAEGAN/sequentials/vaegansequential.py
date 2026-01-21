import numpy as cp
import matplotlib.pyplot as plt
from .basic_sequential import SequentialAnalyzer,SequentialBase
import pickle
try:
    from tqdm import tqdm
except Exception:  # fallback if tqdm isn't installed
    def tqdm(iterable, **kwargs):
        return iterable
 
  

class Encoder(SequentialBase):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x, cond=None):
        x = cp.array(x, dtype=cp.float32)
        if cond is not None:
            cond = cp.array(cond, dtype=cp.float32)
            # x = cp.concatenate([x, cond], axis=1)

        for layer in self.layers:
            x = layer.forward(x, cond=cond) if 'cond' in layer.forward.__code__.co_varnames else layer.forward(x)
        return x

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def update(self, lr=0.001):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(lr) 
 

class Decoder(SequentialBase):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, z, cond=None):
        z = cp.array(z, dtype=cp.float32)
        if cond is not None:
            cond = cp.array(cond, dtype=cp.float32)
            z = cp.concatenate([z, cond], axis=1)

        for layer in self.layers:
            z = layer.forward(z, cond=cond) if 'cond' in layer.forward.__code__.co_varnames else layer.forward(z)
        return z

    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def update(self, lr=0.001):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(lr)
 

class VariationalAutoencoder(SequentialBase, SequentialAnalyzer):
    def __init__(self, encoder, decoder, loss, enable_encoder=True, enable_decoder=True):
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss
        self.enable_encoder = enable_encoder
        self.enable_decoder = enable_decoder

        self.losses = []
        self.main_loss_epoch = []
        self.aditionals_losses_epoch = []
        self.addictional_error = 0

    def forward(self, x, cond=None):
        """Прямой проход через энкодер и декодер"""
        if self.enable_encoder:
            z = self.encoder.forward(x, cond)
        else:
            batch_size = x.shape[0]
            z = cp.random.normal(0, 1, (batch_size, 64)).astype(cp.float32)

        if self.enable_decoder:
            out = self.decoder.forward(z, cond)
        else:
            out = z
        return out

    def backward(self, grad_out):
        """Обратный проход"""
        if self.enable_decoder:
            grad_out = self.decoder.backward(grad_out)
        if self.enable_encoder:
            grad_out = self.encoder.backward(grad_out)
        return grad_out

    def update(self, lr=0.001):
        """Обновление параметров"""
        if self.enable_encoder:
            self.encoder.update(lr)
        if self.enable_decoder:
            self.decoder.update(lr)

    def fit(self, train_x, train_y, train_cond=None, epochs=50, batch_size=64, lr=0.001):
        """Обучение модели"""
        train_x = cp.array(train_x, dtype=cp.float32)
        train_y = cp.array(train_y, dtype=cp.float32)
        if train_cond is not None:
            train_cond = cp.array(train_cond, dtype=cp.float32)

        n_samples = train_x.shape[0]

        for epoch in tqdm(range(epochs), desc="VAE", unit="epoch"):
            idx = cp.random.permutation(n_samples)
            losses = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = train_x[idx[start:end]]
                y_batch = train_y[idx[start:end]]
                cond_batch = train_cond[idx[start:end]] if train_cond is not None else None

                y_pred = self.forward(x_batch, cond_batch)
                loss_value = self.loss.loss_func(y_pred, y_batch)
                grad = self.loss.delta_loss_func(y_pred, y_batch)

                self.backward(grad)
                self.update(lr)
                losses.append(loss_value)

            mean_loss = float(cp.mean(cp.array(losses)))
            self.losses.append(mean_loss)
            print(f"Epoch {epoch}: Loss = {mean_loss:.6f}")

    def show_plots(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.losses, color='blue', lw=2)
        plt.xlabel('Эпоха')
        plt.ylabel('Ошибка')
        plt.title('График ошибки обучения')
        plt.grid(True)
        plt.show(block=True)

    def show_metrics(self):
        print("=== Metrics ===")
        if self.losses:
            print(f"Последняя ошибка: {self.losses[-1]:.6f}")
            print(f"Средняя ошибка: {float(cp.mean(cp.array(self.losses))):.6f}")
        else:
            print("Ошибки отсутствуют (модель не обучалась).")
 
    def toggle_encoder(self, state: bool):
        self.enable_encoder = state

    def toggle_decoder(self, state: bool):
        self.enable_decoder = state



    def save(self, path):
        """Сохранение всех весов энкодера и декодера"""
        weights = {
            "encoder": [],
            "decoder": []
        }

        for layer in self.encoder.layers:
            layer_dict = {}
            if hasattr(layer, "w"): layer_dict["w"] = cp.array(layer.w)
            if hasattr(layer, "b"): layer_dict["b"] = cp.array(layer.b)
            weights["encoder"].append(layer_dict)

        for layer in self.decoder.layers:
            layer_dict = {}
            if hasattr(layer, "w"): layer_dict["w"] = cp.array(layer.w)
            if hasattr(layer, "b"): layer_dict["b"] = cp.array(layer.b)
            weights["decoder"].append(layer_dict)

        with open(path, "wb") as f:
            pickle.dump(weights, f)

        print(f"[VAE] Weights saved to {path}")

    def load(self, path):
        """Загрузка весов в энкодер и декодер"""
        with open(path, "rb") as f:
            weights = pickle.load(f)

        for layer, w in zip(self.encoder.layers, weights["encoder"]):
            if "w" in w and hasattr(layer, "w"):
                layer.w = cp.array(w["w"], dtype=cp.float32)
            if "b" in w and hasattr(layer, "b"):
                layer.b = cp.array(w["b"], dtype=cp.float32)

        for layer, w in zip(self.decoder.layers, weights["decoder"]):
            if "w" in w and hasattr(layer, "w"):
                layer.w = cp.array(w["w"], dtype=cp.float32)
            if "b" in w and hasattr(layer, "b"):
                layer.b = cp.array(w["b"], dtype=cp.float32)

        print(f"[VAE] Weights loaded from {path}")


        
class ConditionalGAN:
    def __init__(self, generator, discriminator, loss, latent_dim, num_classes):
        """
        generator: CVAE-декодер (или любой условный генератор)
        discriminator: дискриминатор
        loss: функция потерь (например BCE)
        latent_dim: размер латентного вектора z
        num_classes: количество классов для условия (one-hot)
        """
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.losses_G = []
        self.losses_D = []


    def generate(self, batch_size, cond):
        """
        cond: one-hot матрица [batch_size, num_classes]
        """
        z = cp.random.normal(0, 1, (batch_size, self.latent_dim)).astype(cp.float32)
        return self.generator.forward(z, cond=cond)


    def train_step(self, real_x, real_cond, lr=0.0002,is_vae_update = True,is_dec_update = True):
        batch_size = real_x.shape[0]

        fake_x = self.generate(batch_size=batch_size,cond=real_cond)  
        pred_fake = self.discriminator.forward(fake_x)
        pred_real = self.discriminator.forward(real_x)
        loss_real = self.loss.loss_func(pred_real, cp.ones_like(pred_real))
        loss_fake = self.loss.loss_func(pred_fake, cp.zeros_like(pred_fake))

        if is_dec_update:
            grad_real = self.loss.delta_loss_func(pred_real, cp.ones_like(pred_real))
            grad_real = self.discriminator.backward(grad_real)

            
            grad_fake = self.loss.delta_loss_func(pred_fake, cp.zeros_like(pred_fake))
            grad_fake = self.discriminator.backward(grad_fake)

            self.discriminator.update(lr)

        if is_vae_update:
            grad = self.loss.delta_loss_func(pred_fake, cp.ones_like(pred_fake))
            grad = self.discriminator.backward(grad)
            grad = self.generator.backward(grad)

            self.discriminator.reset()
            self.generator.update(lr)

        loss_D = float(loss_real + loss_fake)
        loss_G = self.loss.loss_func(pred_fake, cp.ones_like(pred_fake))
        
        return float(loss_G), float(loss_D)

    def fit(self, train_x, train_cond, epochs=50, batch_size=64, lr=0.0002,num_of_learn_gen=5):
        train_x = cp.array(train_x, dtype=cp.float32)
        train_cond = cp.array(train_cond, dtype=cp.float32)
        n_samples = train_x.shape[0]

        for epoch in tqdm(range(epochs), desc="CGAN", unit="epoch"):
            idx = cp.random.permutation(n_samples)
            losses_G = []
            losses_D = []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = train_x[idx[start:end]]
                cond_batch = train_cond[idx[start:end]]

                is_vae_update = epoch % num_of_learn_gen == 0
                is_dec_update = epoch % num_of_learn_gen != 0

                loss_G, loss_D = self.train_step(x_batch, cond_batch, lr,is_vae_update=is_vae_update,is_dec_update=is_dec_update)
                losses_G.append(loss_G)
                losses_D.append(loss_D)

            mean_G = sum(losses_G) / len(losses_G)
            mean_D = sum(losses_D) / len(losses_D)

            self.losses_G.append(mean_G)
            self.losses_D.append(mean_D)

            print(f"Epoch {epoch}:  G = {mean_G:.4f}   D = {mean_D:.4f}")


# vaegansequential.py

class VAEGAN():
    """
    Полноценный VAE-GAN:
    - Энкодер → z (может быть заморожен)
    - Декодер → x̂ (реконструкция + генератор)
    - Дискриминатор отличает настоящие изображения от сгенерированных
    Обучается всё вместе: VAE-реконструкция + KL + GAN-loss на декодере
    """
    def __init__(self, vae, discriminator, latent_dim=64, num_classes=10):
        self.vae = vae                  
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.loss_recon = vae.loss        
        self.loss_gan = discriminator.loss   

        # Логи
        self.losses_total = []
        self.losses_recon = []
        self.losses_gan_g = []
        self.losses_gan_d = []

    def generate(self, batch_size, cond=None):
        z = cp.random.normal(0, 1, (batch_size, self.latent_dim)).astype(cp.float32)
        if cond is not None:
            cond = cp.array(cond, dtype=cp.float32)
        return self.vae.decoder.forward(z, cond=cond)

    def forward(self, x, cond=None):
        return self.vae.forward(x, cond=cond)
 
    def fit(self, train_x, train_cond=None, epochs=100, batch_size=64,
            lr_d=0.0002, lr_g=0.0002, vae_weight=1.0, gan_weight=1.0,
            d_updates_per_step=1,d_steps_before=5):
        train_x = cp.array(train_x, dtype=cp.float32)
        if train_cond is not None:
            train_cond = cp.array(train_cond, dtype=cp.float32)

        n_samples = train_x.shape[0]
        
        for epoch in tqdm(range(epochs), desc="VAEGAN", unit="epoch"):
            idx = cp.random.permutation(n_samples)
            is_vae_enable = epoch>  d_steps_before
            losses_recon, losses_g, losses_d = [], [], []

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = train_x[idx[start:end]]
                cond_batch = train_cond[idx[start:end]] if train_cond is not None else None
                batch_size_cur = x_batch.shape[0]

                for _ in range(d_updates_per_step):
                    z_fake = cp.random.normal(0, 1, (batch_size_cur, self.latent_dim)).astype(cp.float32)
                    x_fake = self.vae.decoder.forward(z_fake, cond=cond_batch)
                    x_fake = x_fake.copy()  

                    d_real = self.discriminator.forward(x_batch)
                    loss_d_real = self.loss_gan.loss_func(d_real, cp.ones_like(d_real))

                    d_fake = self.discriminator.forward(x_fake)
                    loss_d_fake = self.loss_gan.loss_func(d_fake, cp.zeros_like(d_fake))

                    loss_d = (loss_d_real + loss_d_fake) * gan_weight

                    grad = self.loss_gan.delta_loss_func(d_real, cp.ones_like(d_real))
                    self.discriminator.backward(grad)

                    grad = self.loss_gan.delta_loss_func(d_fake, cp.zeros_like(d_fake))
                    self.discriminator.backward(grad)

                    self.discriminator.update(lr_d)

                    losses_d.append(float(loss_d))
                recon_x = self.vae.forward(x_batch, cond=cond_batch)

                loss_recon = self.loss_recon.loss_func(recon_x, x_batch)

                d_fake_for_g = self.discriminator.forward(recon_x)  # НЕ копируем — нужен граф!
                loss_gan_g = self.loss_gan.loss_func(d_fake_for_g, cp.ones_like(d_fake_for_g))

                total_loss_g = vae_weight * loss_recon + gan_weight * loss_gan_g

                grad = self.loss_recon.delta_loss_func(recon_x, x_batch) * vae_weight
                grad_gan = self.loss_gan.delta_loss_func(d_fake_for_g, cp.ones_like(d_fake_for_g)) * gan_weight
                # Backprop through discriminator to get gradient w.r.t. recon_x
                grad_gan = self.discriminator.backward(grad_gan)
                # Avoid leaking discriminator grads into the next D step
                self.discriminator.reset()
                grad += grad_gan
                if is_vae_enable:
                    self.vae.backward(grad)
                    self.vae.update(lr_g)

                losses_recon.append(float(loss_recon))
                losses_g.append(float(loss_gan_g))

            self.losses_recon.append(cp.mean(cp.array(losses_recon)).item())
            self.losses_gan_g.append(cp.mean(cp.array(losses_g)).item())
            self.losses_gan_d.append(cp.mean(cp.array(losses_d)).item())
            self.losses_total.append(cp.mean(cp.array(total_loss_g)).item())
            print(f"Epoch {epoch+1:3d} | Recon: {self.losses_recon[-1]:.5f} | "
                f"GAN_G: {self.losses_gan_g[-1]:.5f} | GAN_D: {self.losses_gan_d[-1]:.5f}")

    def show_plots(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.losses_recon, label="Reconstruction")
        plt.title("MSE Reconstruction Loss")
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(self.losses_gan_g, label="Generator (fool D)", color="green")
        plt.title("GAN Generator Loss")
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.plot(self.losses_gan_d, label="Discriminator", color="red")
        plt.title("GAN Discriminator Loss")
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(self.losses_total, label="Total", color="purple")
        plt.title("Total Loss")
        plt.grid()

        plt.tight_layout()
        plt.show()

