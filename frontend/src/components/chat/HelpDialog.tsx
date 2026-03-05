import { useState } from "react";
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  Rating,
  TextField,
  Snackbar,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import CampaignIcon from "@mui/icons-material/Campaign";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import PollIcon from "@mui/icons-material/Poll";
import PushPinIcon from "@mui/icons-material/PushPin";
import { useQuery, useMutation } from "@tanstack/react-query";
import { contentApi } from "../../api/client";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Announcement {
  id: string;
  title: string;
  content: string;
  is_pinned: boolean;
  is_active: boolean;
  created_at: string;
  start_date?: string;
  end_date?: string;
}

interface FAQ {
  id: string;
  category: string;
  question: string;
  answer: string;
  sort_order: number;
}

interface SurveyQuestion {
  text: string;
  type: "rating" | "single_choice" | "multi_choice" | "text";
  options: string[];
}

interface Survey {
  id: string;
  title: string;
  description: string;
  questions: SurveyQuestion[];
  is_active: boolean;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Announcements Tab
// ---------------------------------------------------------------------------

function AnnouncementsTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["announcements-dialog"],
    queryFn: () => contentApi.listAnnouncements().then((r) => r.data),
    staleTime: 60_000,
  });

  const announcements: Announcement[] = data?.announcements ?? [];

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
        <CircularProgress size={32} />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">공지사항을 불러오지 못했습니다.</Alert>;
  }

  if (announcements.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
        등록된 공지사항이 없습니다.
      </Typography>
    );
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      {announcements.map((ann) => (
        <Box
          key={ann.id}
          sx={{
            border: "1px solid",
            borderColor: ann.is_pinned ? "primary.main" : "divider",
            borderRadius: 2,
            p: 2,
            bgcolor: ann.is_pinned ? "primary.main" + "0d" : "transparent",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1, mb: 0.75 }}>
            {ann.is_pinned && (
              <PushPinIcon sx={{ fontSize: 14, color: "primary.main", mt: 0.25, flexShrink: 0 }} />
            )}
            <Typography variant="subtitle2" fontWeight={600} sx={{ flex: 1 }}>
              {ann.title}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0 }}>
              {new Date(ann.created_at).toLocaleDateString("ko-KR", {
                year: "numeric",
                month: "short",
                day: "numeric",
              })}
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
            {ann.content}
          </Typography>
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// FAQ Tab
// ---------------------------------------------------------------------------

function FAQTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["faq-dialog"],
    queryFn: () => contentApi.listFAQ().then((r) => r.data),
    staleTime: 120_000,
  });

  const faqItems: FAQ[] = data?.faq ?? [];

  // Group by category
  const categories = Array.from(new Set(faqItems.map((f) => f.category)));
  const grouped = categories.map((cat) => ({
    category: cat,
    items: faqItems.filter((f) => f.category === cat),
  }));

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
        <CircularProgress size={32} />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">FAQ를 불러오지 못했습니다.</Alert>;
  }

  if (faqItems.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
        등록된 FAQ가 없습니다.
      </Typography>
    );
  }

  return (
    <Box>
      {grouped.map((group, gi) => (
        <Box key={group.category} sx={{ mb: gi < grouped.length - 1 ? 2 : 0 }}>
          {categories.length > 1 && (
            <Typography
              variant="caption"
              sx={{
                fontWeight: 700,
                fontSize: "0.65rem",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "text.secondary",
                display: "block",
                mb: 0.75,
                px: 0.5,
              }}
            >
              {group.category}
            </Typography>
          )}
          {group.items.map((faq) => (
            <Accordion
              key={faq.id}
              disableGutters
              elevation={0}
              sx={{
                border: "1px solid",
                borderColor: "divider",
                borderRadius: "8px !important",
                mb: 0.75,
                "&:before": { display: "none" },
                "&.Mui-expanded": { borderColor: "primary.main" },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon sx={{ fontSize: 18 }} />}
                sx={{ minHeight: 44, "& .MuiAccordionSummary-content": { my: 1 } }}
              >
                <Typography variant="body2" fontWeight={500}>
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0, pb: 1.5 }}>
                <Divider sx={{ mb: 1.25 }} />
                <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      ))}
    </Box>
  );
}

// ---------------------------------------------------------------------------
// Survey question renderer
// ---------------------------------------------------------------------------

function SurveyQuestionInput({
  question,
  index: _index,
  value,
  onChange,
}: {
  question: SurveyQuestion;
  index: number;
  value: string;
  onChange: (v: string) => void;
}) {
  if (question.type === "rating") {
    const numVal = value ? Number(value) : 0;
    return (
      <Rating
        value={numVal}
        onChange={(_, v) => onChange(String(v ?? 0))}
        size="large"
      />
    );
  }

  if (question.type === "single_choice") {
    return (
      <FormControl>
        <RadioGroup value={value} onChange={(e) => onChange(e.target.value)}>
          {question.options.map((opt) => (
            <FormControlLabel
              key={opt}
              value={opt}
              control={<Radio size="small" />}
              label={<Typography variant="body2">{opt}</Typography>}
            />
          ))}
        </RadioGroup>
      </FormControl>
    );
  }

  if (question.type === "text") {
    return (
      <TextField
        fullWidth
        multiline
        minRows={2}
        size="small"
        placeholder="답변을 입력하세요"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  // multi_choice — simplified as single for now
  return (
    <FormControl>
      <RadioGroup value={value} onChange={(e) => onChange(e.target.value)}>
        {question.options.map((opt) => (
          <FormControlLabel
            key={opt}
            value={opt}
            control={<Radio size="small" />}
            label={<Typography variant="body2">{opt}</Typography>}
          />
        ))}
      </RadioGroup>
    </FormControl>
  );
}

// ---------------------------------------------------------------------------
// Surveys Tab
// ---------------------------------------------------------------------------

function SurveysTab() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["surveys-dialog"],
    queryFn: () => contentApi.listSurveys().then((r) => r.data),
    staleTime: 60_000,
  });

  const surveys: Survey[] = data?.surveys ?? [];

  // Track expanded survey + answers per survey
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [answers, setAnswers] = useState<Record<string, string[]>>({});
  const [submitted, setSubmitted] = useState<Set<string>>(new Set());
  const [snack, setSnack] = useState("");

  const submitMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: unknown[] }) =>
      contentApi.submitSurveyResponse(id, payload),
    onSuccess: (_data, variables) => {
      setSubmitted((prev) => new Set(prev).add(variables.id));
      setSnack("설문에 응답해 주셔서 감사합니다!");
    },
    onError: () => {
      setSnack("설문 제출에 실패했습니다.");
    },
  });

  const handleAnswer = (surveyId: string, qIndex: number, value: string) => {
    setAnswers((prev) => {
      const surveyAnswers = [...(prev[surveyId] ?? [])];
      surveyAnswers[qIndex] = value;
      return { ...prev, [surveyId]: surveyAnswers };
    });
  };

  const handleSubmit = (survey: Survey) => {
    const surveyAnswers = answers[survey.id] ?? [];
    const payload = survey.questions.map((_, qi) => ({
      question_index: qi,
      value: surveyAnswers[qi] ?? "",
    }));
    submitMutation.mutate({ id: survey.id, payload });
  };

  if (isLoading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
        <CircularProgress size={32} />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">설문을 불러오지 못했습니다.</Alert>;
  }

  if (surveys.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ textAlign: "center", py: 4 }}>
        진행 중인 설문이 없습니다.
      </Typography>
    );
  }

  return (
    <>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
        {surveys.map((survey) => {
          const isExpanded = expandedId === survey.id;
          const isDone = submitted.has(survey.id);
          return (
            <Box
              key={survey.id}
              sx={{
                border: "1px solid",
                borderColor: isExpanded ? "primary.main" : "divider",
                borderRadius: 2,
                overflow: "hidden",
              }}
            >
              {/* Header */}
              <Box
                onClick={() => !isDone && setExpandedId(isExpanded ? null : survey.id)}
                sx={{
                  p: 2,
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  cursor: isDone ? "default" : "pointer",
                  "&:hover": !isDone ? { bgcolor: "action.hover" } : {},
                }}
              >
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle2" fontWeight={600}>
                    {survey.title}
                  </Typography>
                  {survey.description && (
                    <Typography variant="caption" color="text.secondary">
                      {survey.description}
                    </Typography>
                  )}
                </Box>
                {isDone ? (
                  <Chip label="응답 완료" size="small" color="success" />
                ) : (
                  <Chip
                    label={`${survey.questions.length}문항`}
                    size="small"
                    color={isExpanded ? "primary" : "default"}
                  />
                )}
              </Box>

              {/* Questions */}
              {isExpanded && !isDone && (
                <>
                  <Divider />
                  <Box sx={{ p: 2, display: "flex", flexDirection: "column", gap: 2.5 }}>
                    {survey.questions.map((q, qi) => (
                      <Box key={qi}>
                        <Typography variant="body2" fontWeight={500} sx={{ mb: 0.75 }}>
                          {qi + 1}. {q.text}
                        </Typography>
                        <SurveyQuestionInput
                          question={q}
                          index={qi}
                          value={(answers[survey.id] ?? [])[qi] ?? ""}
                          onChange={(v) => handleAnswer(survey.id, qi, v)}
                        />
                      </Box>
                    ))}
                    <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                      <Button
                        variant="contained"
                        size="small"
                        disableElevation
                        disabled={submitMutation.isPending}
                        onClick={() => handleSubmit(survey)}
                      >
                        {submitMutation.isPending ? "제출 중..." : "제출"}
                      </Button>
                    </Box>
                  </Box>
                </>
              )}
            </Box>
          );
        })}
      </Box>

      <Snackbar
        open={!!snack}
        autoHideDuration={3000}
        onClose={() => setSnack("")}
        message={snack}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      />
    </>
  );
}

// ---------------------------------------------------------------------------
// HelpDialog (main export)
// ---------------------------------------------------------------------------

interface HelpDialogProps {
  open: boolean;
  onClose: () => void;
}

export function HelpDialog({ open, onClose }: HelpDialogProps) {
  const [tab, setTab] = useState(0);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{ sx: { borderRadius: 3, maxHeight: "80vh" } }}
    >
      <DialogTitle sx={{ pb: 0 }}>
        <Typography variant="h6" fontWeight={700}>
          도움말 및 공지
        </Typography>
      </DialogTitle>

      <Box sx={{ px: 3 }}>
        <Tabs
          value={tab}
          onChange={(_, v) => setTab(v)}
          variant="fullWidth"
          sx={{ borderBottom: "1px solid", borderColor: "divider" }}
        >
          <Tab
            icon={<CampaignIcon sx={{ fontSize: 16 }} />}
            iconPosition="start"
            label="공지사항"
            sx={{ minHeight: 48, fontSize: "0.8rem", textTransform: "none" }}
          />
          <Tab
            icon={<HelpOutlineIcon sx={{ fontSize: 16 }} />}
            iconPosition="start"
            label="FAQ"
            sx={{ minHeight: 48, fontSize: "0.8rem", textTransform: "none" }}
          />
          <Tab
            icon={<PollIcon sx={{ fontSize: 16 }} />}
            iconPosition="start"
            label="설문"
            sx={{ minHeight: 48, fontSize: "0.8rem", textTransform: "none" }}
          />
        </Tabs>
      </Box>

      <DialogContent sx={{ pt: 2 }}>
        {tab === 0 && <AnnouncementsTab />}
        {tab === 1 && <FAQTab />}
        {tab === 2 && <SurveysTab />}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose} variant="outlined" size="small">
          닫기
        </Button>
      </DialogActions>
    </Dialog>
  );
}
