
sed -i -e  "s/\('pred_grid':\).*/\1 2,/" -e  "s/\('num_agents':\).*/\1 1,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_2_1.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 2,/" -e  "s/\('num_agents':\).*/\1 2,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_2_2.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 2,/" -e  "s/\('num_agents':\).*/\1 3,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_2_3.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 3,/" -e  "s/\('num_agents':\).*/\1 1,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_3_1.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 3,/" -e  "s/\('num_agents':\).*/\1 2,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_3_2.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 3,/" -e  "s/\('num_agents':\).*/\1 3,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_3_3.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 6,/" -e  "s/\('num_agents':\).*/\1 1,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_6_1.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 6,/" -e  "s/\('num_agents':\).*/\1 2,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_6_2.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 6,/" -e  "s/\('num_agents':\).*/\1 3,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_6_3.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 12,/" -e  "s/\('num_agents':\).*/\1 1,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_12_1.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 12,/" -e  "s/\('num_agents':\).*/\1 2,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_12_2.log &
sleep 10

sed -i -e  "s/\('pred_grid':\).*/\1 12,/" -e  "s/\('num_agents':\).*/\1 3,/" para.py
nohup python mdp_map_agent.py &> log/mdp_m_12_3.log &
