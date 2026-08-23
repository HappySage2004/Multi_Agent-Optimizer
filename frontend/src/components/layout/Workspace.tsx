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
import { Sidebar } from "@/components/layout/Sidebar";
import { TopHeader } from "@/components/layout/TopHeader";
import { useCampaignRun } from "@/hooks/useCampaignRun";
import { provenanceInfo } from "@/lib/derive";

export function Workspace() {
  const campaign = useCampaignRun();
  const [activeTab, setActiveTab] = useState<InspectorTabId>("d1");

  const run = campaign.runData.run;
  const hasPackage = Boolean(run?.optimization?.package);
  const provenance = provenanceInfo(run).provenance;

  const title =
    run?.campaign_spec.campaign_objective ?? campaign.activeSession?.title ?? "New Campaign";

  /** The strategy card's "Inspector Panel" link jumps to the optimizer view. */
  const openInspector = useCallback(() => setActiveTab("d4"), []);

  const exportProposal = useCallback(() => {
    // The proposal PDF is generated from the run record; print-to-PDF is the honest
    // interim behaviour rather than a button that silently does nothing.
    window.print();
  }, []);

  return (
    <ResizableLayout
      sidebar={
        <Sidebar
          sessions={campaign.sessions}
          activeSessionId={campaign.activeSessionId}
          health={campaign.health}
          onNewCampaign={() => void campaign.newCampaign()}
          onSelectSession={campaign.selectSession}
          onDeleteSession={(id) => void campaign.removeSession(id)}
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
  );
}
