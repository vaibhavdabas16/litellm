import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LensPage from "./page";

const { auth, workspace } = vi.hoisted(() => ({ auth: vi.fn(), workspace: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: auth }));
vi.mock("@/components/lens/EmbeddedLens", () => ({
  EmbeddedLens: () => {
    workspace();
    return <div>Lens workspace</div>;
  },
}));

describe("Lens route", () => {
  beforeEach(() => vi.clearAllMocks());

  it("waits for an access token", () => {
    auth.mockReturnValue({ accessToken: null, isAuthorized: false, isLoading: false });
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
    expect(workspace).not.toHaveBeenCalled();
  });

  it("shows the Lens page once the dashboard is signed in", () => {
    auth.mockReturnValue({ accessToken: "test-token", isAuthorized: true, isLoading: false });
    render(<LensPage />);
    expect(screen.getByText("Lens workspace")).toBeVisible();
  });

  it.each([
    { isAuthorized: true, isLoading: true },
    { isAuthorized: false, isLoading: false },
  ])("keeps Lens closed with authorization state $isAuthorized and loading state $isLoading", (session) => {
    auth.mockReturnValue({ accessToken: "test-token", ...session });
    render(<LensPage />);
    expect(screen.queryByText("Lens workspace")).not.toBeInTheDocument();
  });
});
