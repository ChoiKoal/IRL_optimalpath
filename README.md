# IRL_optimalpath_MC_trial

# main함수를 term_project라는 함수로 바꿈
# 필요하다고 생각되는 변수들은 term_project 함수 내부의 local 변수로 바꿈(interation수, trajectory 개수, height, width 등)

# meeting 때 이야기 한 것처럼 총 4개의 obstacle option에 대한 실험이 구현되어있음(이 환경을 만드는 함수는 util.py의 맨 마지막 부분에 있음)
	1. 항상 하던 obstacle
	2. 새로 만든 obstacle
	3. 1번 환경 + random obstacle(random 개수는 size별로 5일때 5개, 7일때 10개, 9일때 20개로 했음, 내 맘대로임)
	4. No obstacle + random reward

# 현재버전은 obstacle option만 MC trial에 적용되는 control variable이며 iteration, trajectory 개수를 바꿔가면서 MC trial을 하는 것은 필요하다고 판단되면 구현할 것

# 실험을 돌릴 때는 gw_size 변수만 5, 7, 9를 선택하면 됨

# 실험의 산출물은 하나의 gridworld size에 대해 4개의 obstacle option을 적용하여 실행한 결과가 그림파일로 저장되고 command line에 sum of reward diffrence가 출력됨
# 저장되는 그림파일은 각 MC trial별 맨날 봤던 그림, MC trial별 sum of reward difference 변화 그리고 obstacle option별 sum of reward difference의 평균값(MC_trial 평균)임.

# MC_Number를 100으로 하면 시간이 오래걸리니 일단 1로 설정하고 각자 생각한 알고리즘을 적용해서 실험해보는게 좋을 것 같음

# 그리고 화요일 미팅결과로 알고리즘 최종 확정하면 그때 MC_Number 100으로 해서 돌리면 될 것 같음(5, 7, 9 size를 하나씩 맡아서 돌리면 되지 않을까 싶음)


