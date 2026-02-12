import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Box, Button, Card, CardContent, Chip, Dialog, DialogTitle, DialogContent,
  DialogActions, Grid, IconButton, Tab, Tabs, Table, TableBody, TableCell,
  TableContainer, TableHead, TableRow, Paper, TextField, Typography,
  CircularProgress, Switch, FormControlLabel,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import PushPinIcon from "@mui/icons-material/PushPin";
import { contentApi } from "../api/client";

// ── Announcements Tab ──
function AnnouncementsTab() {
  const qc = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [pinned, setPinned] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: ["announcements"],
    queryFn: () => contentApi.listAnnouncements(),
  });

  const createMut = useMutation({
    mutationFn: () => contentApi.createAnnouncement({ title, content, is_pinned: pinned }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["announcements"] }); setDialogOpen(false); setTitle(""); setContent(""); },
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => contentApi.deleteAnnouncement(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["announcements"] }),
  });

  const items = data?.data?.announcements ?? [];

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>공지사항 관리</Typography>
        <Button size="small" variant="contained" startIcon={<AddIcon />} onClick={() => setDialogOpen(true)}>추가</Button>
      </Box>
      {isLoading ? <CircularProgress /> : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>제목</TableCell>
                <TableCell>상태</TableCell>
                <TableCell>작성일</TableCell>
                <TableCell align="right">작업</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {items.map((a: any) => (
                <TableRow key={a.id}>
                  <TableCell>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      {a.is_pinned && <PushPinIcon fontSize="small" color="primary" />}
                      <Typography variant="body2">{a.title}</Typography>
                    </Box>
                  </TableCell>
                  <TableCell><Chip label={a.is_active ? "활성" : "비활성"} size="small" color={a.is_active ? "success" : "default"} /></TableCell>
                  <TableCell><Typography variant="caption">{new Date(a.created_at).toLocaleDateString("ko-KR")}</Typography></TableCell>
                  <TableCell align="right">
                    <IconButton size="small" onClick={() => deleteMut.mutate(a.id)}><DeleteIcon fontSize="small" /></IconButton>
                  </TableCell>
                </TableRow>
              ))}
              {items.length === 0 && <TableRow><TableCell colSpan={4} align="center"><Typography variant="body2" color="text.secondary">공지사항이 없습니다.</Typography></TableCell></TableRow>}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>공지사항 추가</DialogTitle>
        <DialogContent>
          <TextField fullWidth label="제목" value={title} onChange={e => setTitle(e.target.value)} sx={{ mt: 1, mb: 2 }} size="small" />
          <TextField fullWidth label="내용" value={content} onChange={e => setContent(e.target.value)} multiline rows={4} size="small" />
          <FormControlLabel control={<Switch checked={pinned} onChange={e => setPinned(e.target.checked)} />} label="상단 고정" sx={{ mt: 1 }} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={() => createMut.mutate()} disabled={!title}>저장</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// ── FAQ Tab ──
function FAQTab() {
  const qc = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [category, setCategory] = useState("일반");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["faq"],
    queryFn: () => contentApi.listFAQ(),
  });

  const createMut = useMutation({
    mutationFn: () => contentApi.createFAQ({ category, question, answer }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["faq"] }); setDialogOpen(false); setQuestion(""); setAnswer(""); },
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => contentApi.deleteFAQ(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["faq"] }),
  });

  const items = data?.data?.faq ?? [];

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>FAQ 관리</Typography>
        <Button size="small" variant="contained" startIcon={<AddIcon />} onClick={() => setDialogOpen(true)}>추가</Button>
      </Box>
      {isLoading ? <CircularProgress /> : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>카테고리</TableCell>
                <TableCell>질문</TableCell>
                <TableCell>답변</TableCell>
                <TableCell align="right">작업</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {items.map((f: any) => (
                <TableRow key={f.id}>
                  <TableCell><Chip label={f.category} size="small" /></TableCell>
                  <TableCell><Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>{f.question}</Typography></TableCell>
                  <TableCell><Typography variant="body2" noWrap sx={{ maxWidth: 300 }}>{f.answer}</Typography></TableCell>
                  <TableCell align="right">
                    <IconButton size="small" onClick={() => deleteMut.mutate(f.id)}><DeleteIcon fontSize="small" /></IconButton>
                  </TableCell>
                </TableRow>
              ))}
              {items.length === 0 && <TableRow><TableCell colSpan={4} align="center"><Typography variant="body2" color="text.secondary">FAQ가 없습니다.</Typography></TableCell></TableRow>}
            </TableBody>
          </Table>
        </TableContainer>
      )}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>FAQ 추가</DialogTitle>
        <DialogContent>
          <TextField fullWidth label="카테고리" value={category} onChange={e => setCategory(e.target.value)} sx={{ mt: 1, mb: 2 }} size="small" />
          <TextField fullWidth label="질문" value={question} onChange={e => setQuestion(e.target.value)} sx={{ mb: 2 }} size="small" />
          <TextField fullWidth label="답변" value={answer} onChange={e => setAnswer(e.target.value)} multiline rows={3} size="small" />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>취소</Button>
          <Button variant="contained" onClick={() => createMut.mutate()} disabled={!question || !answer}>저장</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// ── Surveys Tab ──
function SurveysTab() {
  const { data, isLoading } = useQuery({
    queryKey: ["surveys"],
    queryFn: () => contentApi.listSurveys(),
  });

  const items = data?.data?.surveys ?? [];

  return (
    <Box>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2 }}>설문 관리</Typography>
      {isLoading ? <CircularProgress /> : (
        <Grid container spacing={2}>
          {items.map((s: any) => (
            <Grid key={s.id} size={{ xs: 12, sm: 6 }}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{s.title}</Typography>
                  <Typography variant="caption" color="text.secondary">{s.description}</Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>질문 수: {s.questions?.length ?? 0}</Typography>
                  <Chip label={s.is_active ? "진행 중" : "종료"} size="small" color={s.is_active ? "success" : "default"} sx={{ mt: 1 }} />
                </CardContent>
              </Card>
            </Grid>
          ))}
          {items.length === 0 && <Grid size={{ xs: 12 }}><Typography color="text.secondary" variant="body2">등록된 설문이 없습니다.</Typography></Grid>}
        </Grid>
      )}
    </Box>
  );
}

// ── Main Component ──
export default function ContentManager() {
  const [tab, setTab] = useState(0);

  return (
    <Box>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="공지사항" />
        <Tab label="FAQ" />
        <Tab label="설문" />
      </Tabs>
      {tab === 0 && <AnnouncementsTab />}
      {tab === 1 && <FAQTab />}
      {tab === 2 && <SurveysTab />}
    </Box>
  );
}
