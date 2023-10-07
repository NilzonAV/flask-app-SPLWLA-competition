function  [pickA1,pickBB,flags] = sxk_RefineCorrection_GR_RD_head(a1,b1,pickA0,s1_scope,p1_refine,tensile_pool,pad_numval)

pickB0 = pickA0;  %% because a basic shift has been done already 
a1log = a1.trend + a1.detail; % 
a2log = a1.trend + a1.detail;
b1log = b1.trend + b1.detail; % 
b2log = b1.trend + b1.detail ;

idxvec_ori = [1:1:length(a1log)].';
jud_loga = a1.trend~=pad_numval | a1log~=2.*pad_numval;
jud_logb = b1.trend~=pad_numval | b1log~=2.*pad_numval;

pickBB = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% S1 : determine the A-picks for comparison 
a1log_eff = a1log(jud_loga);
a2log_eff = a2log(jud_loga);
b1log_eff   = b1log(jud_logb);
b2log_eff   = b2log(jud_logb);
idxvec_Aeff = idxvec_ori(jud_loga);
idxvec_Beff = idxvec_ori(jud_logb);

alog_idxmin = min(idxvec_Aeff);
alog_idxmax = max(idxvec_Aeff);
blog_idxmin = min(idxvec_Beff);
blog_idxmax = max(idxvec_Beff);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nrad_retune  = 80.*s1_scope; %% log segments for comparsion
min_coverlen = 2.*s1_scope; 

space_L1 = 8;

pickA1    = sub_nearestPick_NODC(a1log_eff,idxvec_Aeff,pickA0);
pickB1set = sub_deterimineBpicks(pickA1,p1_refine,idxvec_Beff,b1log_eff,space_L1);
pickB1set = pickB1set(pickB1set>=blog_idxmin & pickB1set<=blog_idxmax);

idxaa = pickA0 + [-nrad_retune:1:nrad_retune];
idxbb = pickA0 + [-nrad_retune:1:nrad_retune]; %% because a basic shift has been done already
judaae = idxaa>=alog_idxmin & idxaa<=alog_idxmax;
judbbe = idxbb>=blog_idxmin & idxbb<=blog_idxmax;
idxaae = idxaa(judaae);
idxbbe = idxbb(judbbe);

[~,pickA1_rel] = min(abs(idxaae-pickA1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Level 1 : using trend information (a1log & b1log) to determine the picks 
loga1_tar     = a1log(idxaae);
logb1_tar     = b1log(idxbbe);

[xx,yy] = meshgrid(idxbbe,pickB1set);
[~,pickB1s_rel] = min(abs(xx-yy),[],2);
% pickB1s_rel   = pickA1_rel + (pickB1set-pickB0);

obj2d   = sub_ObjFun_LaterPart_retune_GR_RD(loga1_tar,logb1_tar,pickA1_rel,pickB1s_rel,tensile_pool,min_coverlen);

[maxval,maxidx1d] = max(obj2d(:));
[maxridx,maxcidx] = ind2sub(size(obj2d),maxidx1d);
obj2d_mf          = medfilt2(obj2d,[11 11]);
flag1              = (abs(maxval-obj2d_mf(maxidx1d)))./(obj2d_mf(maxidx1d));%%% 

pickB1 = pickB1set(maxcidx);
pickBB = [pickBB pickB1];

str1 =  ['A : ', num2str(pickA1) , ' ~ B : ',num2str(pickB1)];
figure(10002)
clf;
subplot(2,10,1:2);
imagesc(pickB1set,tensile_pool*100,obj2d);
xlabel('B-pick index');
ylabel('Tensile Factor, %')
title(num2str(flag1*100));

subplot(2,10,3:10);
plot(idxvec_Aeff,a1log_eff,'-');
hold on;
plot(idxvec_Beff,b1log_eff,'-');
plot(pickA1,a1log(pickA1),'r*');
plot(pickA1,a1log(pickA1),'ro');
plot(pickB1,b1log(pickB1),'b*');
plot(pickB1,b1log(pickB1),'bo');
grid on;
grid minor;
title(['(pass01) ',str1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Level 2 : using DETAIL information (a2log & b2log) to determine the picks 
space_L2 = 1;
pickB2set = sub_deterimineBpicks(pickB1,space_L1*2,idxvec_Beff,b1log_eff+b2log_eff,space_L2);

loga2_tar     = a2log(idxaae);
logb2_tar     = b2log(idxbbe);


% pickB2s_rel   = pickA1_rel + (pickB2set-pickB0);
[xx,yy] = meshgrid(idxbbe,pickB2set);
[~,pickB2s_rel] = min(abs(xx-yy),[],2);
obj2d   = sub_ObjFun_LaterPart_retune_GR_RD(loga2_tar,logb2_tar,pickA1_rel,pickB2s_rel,tensile_pool,min_coverlen);


[maxval,maxidx1d] = max(obj2d(:));
[maxridx,maxcidx] = ind2sub(size(obj2d),maxidx1d);
obj2d_mf          = medfilt2(obj2d,[11 11]);
flag2              = (abs(maxval-obj2d_mf(maxidx1d)))./(obj2d_mf(maxidx1d));%%% 

pickB2 = pickB2set(maxcidx);
pickBB = [pickBB pickB2];

str2 =  ['A : ', num2str(pickA1) , ' ~ B : ',num2str(pickB2)];
figure(10002)
subplot(2,10,11:12);
imagesc(pickB2set,tensile_pool*100,obj2d);
xlabel('B-pick index');
ylabel('Tensile Factor, %')
title(num2str(flag2*100));
subplot(2,10,13:20);
plot(idxvec_Aeff,a1log_eff,'-');
hold on;
plot(idxvec_Beff,b1log_eff,'-');
plot(pickA1,a1log(pickA1),'r*');
plot(pickA1,a1log(pickA1),'ro');
plot(pickB2,b1log(pickB2),'b*');
plot(pickB2,b1log(pickB2),'bo');
grid on;
grid minor;
title(['(pass02) ',str2]);

flags = [flag1 flag2];


return;


end


function pickB1set = sub_deterimineBpicks(pickA1,p1_refine,idxvec_Beff,b1log_eff,space)

    jud_Bpicks  = idxvec_Beff >= (pickA1-p1_refine) & idxvec_Beff <= (pickA1+p1_refine);
    pickB0set   = idxvec_Beff(jud_Bpicks);
    pickB0set   = [min(pickB0set):space:max(pickB0set)];  %%%% min space : 4 (can affect the accuracy)
    pickB1set   = sub_nearestPick_NODC(b1log_eff,idxvec_Beff,pickB0set);

return;


end


function pickA1 = sub_nearestPick_NODC(a1log_eff,idxvec_Aeff,pickA0)

tmp_log = diff(a1log_eff(:),1,1);
jud_log = tmp_log~=0;
tmp_idx = idxvec_Aeff(1:end-1);
tmp_idx = tmp_idx(jud_log);


pickA1 = [];
for ip = 1:1:length(pickA0)
    [~,idx] = min(abs(tmp_idx-pickA0(ip)));
    pickA1 = [pickA1;tmp_idx(idx)];    
end

return;

end
