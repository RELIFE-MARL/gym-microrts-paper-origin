# download models
python download_models.py --wandb-project vwxyzjn/gym-microrts-paper

# best seed evals
# the best seed is selected according to 
# https://wandb.ai/vwxyzjn/gym-microrts-paper/reports/Selecting-the-best-seed--Vmlldzo0OTY1MjU
python agent_eval.py --exp-name ppo_coacai --agent-model-path trained_models/ppo_coacai/agent-2.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_coacai_no_mask --agent-model-path trained_models/ppo_coacai_no_mask/agent-3.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_diverse --agent-model-path trained_models/ppo_diverse/agent-4.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_diverse_impala --agent-model-path trained_models/ppo_diverse_impala/agent-2.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_gridnet_diverse --agent-model-path trained_models/ppo_gridnet_diverse/agent-1.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_gridnet_diverse_impala --agent-model-path trained_models/ppo_gridnet_diverse_impala/agent-3.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_gridnet_naive --agent-model-path trained_models/ppo_gridnet_naive/agent-2.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_gridnet_diverse_encode_decode --agent-model-path trained_models/ppo_gridnet_diverse_encode_decode/agent-1.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_gridnet_coacai --agent-model-path trained_models/ppo_gridnet_coacai/agent-2.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_coacai_partial_mask --agent-model-path trained_models/ppo_coacai_partial_mask/agent-4.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name ppo_coacai_naive --agent-model-path trained_models/ppo_coacai_naive/agent-3.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name  ppo_gridnet_coacai_naive --agent-model-path trained_models/ppo_gridnet_coacai_naive/agent-1.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name  ppo_gridnet_coacai_partial_mask --agent-model-path trained_models/ppo_gridnet_coacai_partial_mask/agent-1.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity vwxyzjn --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video
python agent_eval.py --exp-name  ppo_gridnet_coacai_no_mask --agent-model-path trained_models/ppo_gridnet_coacai_no_mask/agent-1.pt \
    --max-steps 4000 --num-eval-runs 100 --wandb-entity thinh-huynh-re --wandb-project-name gym-microrts-paper-eval --prod-mode --capture-video

