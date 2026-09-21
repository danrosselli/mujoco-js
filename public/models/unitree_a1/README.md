# Policy de locomoção (RL) do Unitree A1

O viewer (Three.js + MuJoCo WASM) roda uma policy de rede neural exportada para
ONNX e aplica as ações nos 12 atuadores de posição. Nada de treino acontece no
navegador — só a **inferência**.

## Como usar

1. Coloque o checkpoint treinado aqui como:

   ```
   public/models/unitree_a1/policy.onnx
   ```

2. Ajuste [`policy.json`](policy.json) para casar com o checkpoint.
3. `bun run dev` — se o `.onnx` não existir, o viewer continua funcionando na
   pose `home` (e mostra o aviso no painel).

## Contrato da policy

Por padrão seguimos a convenção do `legged_gym` / `unitree_rl_gym` para o A1:
12 juntas, entrada `obs` (1×45 float32), saída (1×12).

Observação (45 dims), na ordem:

| Segmento  | Dims | Conteúdo                                                    | Escala (`obsScales`) |
|-----------|------|-------------------------------------------------------------|----------------------|
| `angVel`  | 3    | velocidade angular do tronco (frame do corpo)               | 0.25                 |
| `gravity` | 3    | gravidade projetada no frame do corpo (≈ `Rᵀ·(0,0,−1)`)     | 1                    |
| `command` | 3    | `[vx, vy, yaw]` alvo                                        | 2.0 / 2.0 / 0.25     |
| `dofPos`  | 12   | `q − defaultJointAngles`                                    | 1.0                  |
| `dofVel`  | 12   | velocidade das juntas                                       | 0.05                 |
| `actions` | 12   | ação do passo anterior                                      | 1                    |

Ação → atuador: `ctrl[i] = defaultJointAngles[i] + action[i] * actionScale`.

`jointOrder` define a ordem das 12 juntas que a policy espera (por nome do
joint no MuJoCo). O controlador acha o atuador correspondente automaticamente,
então a ordem dos atuadores no XML não importa.

## Ajustando para outro checkpoint

- **Outro conjunto de observações** (ex.: incluir `linVel`, ou varredura de
  altura): mude `obsOrder` e as escalas. Segmentos disponíveis:
  `angVel`, `linVel`, `gravity`, `command`, `dofPos`, `dofVel`, `actions`.
- **Outra ordem de juntas**: reordene `jointOrder`.
- **Outra escala de ação**: mude `actionScale`.
- **Outra frequência de controle**: mude `controlHz` (o viewer decima a física
  automaticamente conforme `model.opt.timestep`).

## Armadilhas comuns

- **Convenção de frame**: assumimos `qvel` do freejoint com linear no frame
  global e angular no frame do corpo (padrão do MuJoCo). Se o seu env treinou
  com outra convenção, o `linVel`/`angVel` pode precisar de ajuste.
- **Escalas**: escalas erradas fazem o robô cair na primeira ação. Confira com o
  `policy.json` de referência do repositório de treino.
- **Ordem das pernas**: `FL, FR, RL, RR` é o padrão do `legged_gym`; confira no
  seu checkpoint.