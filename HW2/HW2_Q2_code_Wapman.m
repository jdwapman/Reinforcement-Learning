value = 0;
gamma = 0.9;

for k = 0:1:10000
   value = value + gamma^(5*k)*10;
end

value