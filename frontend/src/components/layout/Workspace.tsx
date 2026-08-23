"use client";

/**
 * The workspace root: owns the run state machine and hands each panel what it needs.
 */

import { useCallback, useState } from "react";

import { ChatFeed } from "@/components/chat/ChatFeed";
import { PromptInputBar } from "@/components/chat/PromptInputBar";
import {
  type InspectorTabId,
  InspectorPanel,
} from "@/components/inspector/InspectorPanel";
import { ResizableLayout } from "@/components/layout/ResizableLayout";
import { EmailDialog } from "@/components/proposal/EmailDialog";
import { ProposalDocument } from "@/components/proposal/ProposalDocument";
import { SettingsDialog } from "@/components/layout/SettingsDialog";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopHeader } from "@/components/layout/TopHeader";
import { useCampaignRun } from "@/hooks/useCampaignRun";
import { packageMetrics, provenanceInfo } from "@/lib/derive";

export function Workspace() {
  const campaign = useCampaignRun();
  const [activeTab, setActiveTab] = useState<InspectorTabId>("d1");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [emailOpen, setEmailOpen] = useState(false);

  const run = campaign.runData.run;
  const pkg = run?.optimization?.package ?? null;
  const hasPackage = Boolean(pkg);
  const provenance = provenanceInfo(run).provenance;

  const title =
    run?.campaign_spec.campaign_objective ?? campaign.activeSession?.title ?? "New Campaign";

  /** The strategy card's "Inspector Panel" link jumps to the optimizer view. */
  const openInspector = useCallback(() => setActiveTab("d4"), []);

  // The printed proposal and the email draft both read these, so a client can never
  // receive a figure the package does not contain.
  const metrics =
    run && pkg
      ? packageMetrics(pkg, run.campaign_spec, campaign.runData.packagedCandidates)
      : null;

  const exportProposal = useCallback(() => {
    // `ProposalDocument` is the only thing visible under @media print, so this prints the
    // client proposal rather than the workspace. Print-to-PDF is the honest interim
    // behaviour; a server-rendered PDF would need a headless browser on the backend.
    window.print();
  }, []);

  return (
    <>
      <ResizableLayout
        sidebar={
          <Sidebar
            sessions={campaign.sessions}
            activeSessionId={campaign.activeSessionId}
            health={campaign.health}
            modelSelection={campaign.modelSelection}
            onNewCampaign={() => void campaign.newCampaign()}
            onSelectSession={campaign.selectSession}
            onDeleteSession={(id) => void campaign.removeSession(id)}
            onOpenSettings={() => setSettingsOpen(true)}
          />
        }
        center={
          <main className="relative flex min-w-0 flex-1 flex-col justify-between bg-white">
            <TopHeader
              title={title}
              status={campaign.status}
              provenance={provenance}
              hasPackage={hasPackage}
              onReset={campaign.resetTranscript}
              onExport={exportProposal}
              onEmail={() => setEmailOpen(true)}
            />

            <ChatFeed
              messages={campaign.messages}
              runData={campaign.runData}
              status={campaign.status}
              stages={campaign.stageStates}
              toolTrail={campaign.toolTrail}
              error={campaign.error}
              pendingQuestions={campaign.pendingQuestions}
              onCancel={campaign.cancel}
              onOpenInspector={openInspector}
              onDismissError={campaign.dismissError}
              onAnswerClarification={(reply) => void campaign.answerClarification(reply)}
            />

            <PromptInputBar
              busy={campaign.status === "streaming"}
              hasPackage={hasPackage}
              pendingUploads={campaign.pendingUploads}
              onSubmit={(query) => void campaign.submit(query)}
              onAttach={(file) => void campaign.attachFile(file)}
              onRemoveUpload={campaign.removePendingUpload}
            />
          </main>
        }
        inspector={
          <InspectorPanel
            runData={campaign.runData}
            activeTab={activeTab}
            onTabChange={setActiveTab}
          />
        }
      />

      {/* Rendered as a sibling of the layout, not inside a panel: the overlay is
        position-fixed and a resizable column with `overflow-hidden` would clip it. */}
      {/* Off-screen on screen, the whole page when printing. Mounted whenever a package
        exists so Ctrl+P works without pressing Export first. */}
      {run && pkg && metrics ? (
        <ProposalDocument
          pkg={pkg}
          spec={run.campaign_spec}
          candidates={campaign.runData.packagedCandidates}
          metrics={metrics}
          title={title}
        />
      ) : null}

      {emailOpen && run && pkg && metrics ? (
        <EmailDialog
          pkg={pkg}
          spec={run.campaign_spec}
          candidates={campaign.runData.packagedCandidates}
          metrics={metrics}
          title={title}
          onClose={() => setEmailOpen(false)}
        />
      ) : null}

      <SettingsDialog
        open={settingsOpen}
        models={campaign.models}
        selection={campaign.modelSelection}
        loadError={campaign.modelsError}
        onSelect={campaign.selectModel}
        onClose={() => setSettingsOpen(false)}
      />
    </>
  );
}
